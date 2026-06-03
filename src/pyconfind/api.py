"""High-level library API.

For library use (notebooks, pipelines, scripts), call :func:`analyze`. It
takes a PDB path and returns the :class:`Analysis` result with all the
pieces — surviving rotamers per position, contact-degree report, and the
loaded library — so downstream code can format, slice, or compute further
without re-loading anything.

Example
-------
>>> from pyconfind import analyze
>>> result = analyze("input.pdb")  # library auto-downloaded + cached
>>> result.contacts_dataframe().nlargest(10, "degree")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .build import PositionRotamers, build_position_rotamers
from .contacts import ContactReport, compute_contacts
from .data import cached_rotamer_library
from .pdb import atoms_from_gemmi_structure, filter_atoms_by_position, read_structure
from .rotlib import RotamerLibrary, load_library
from .selection import select_residue_mask
from .structure import positions_from_atoms

if TYPE_CHECKING:
    import pandas as pd


def _select_contact_backend(backend: str) -> Callable[..., ContactReport]:
    """Return the contact-degree function for the requested backend.

    ``"auto"`` prefers the Numba backend and silently falls back to the pure
    Python reference if Numba (or llvmlite) is unavailable.
    """
    if backend not in ("auto", "numba", "python"):
        raise ValueError(f"unknown backend {backend!r}; use auto/numba/python")
    if backend == "python":
        return compute_contacts
    try:
        from .contacts_numba import compute_contacts_numba
    except ImportError:
        if backend == "numba":
            raise
        return compute_contacts
    return compute_contacts_numba


@dataclass(frozen=True)
class Analysis:
    """Result of a complete confind run on one structure.

    Attributes
    ----------
    positions
        One :class:`PositionRotamers` per residue. Per-position bookkeeping
        (rotamers that survived pruning, fraction pruned, permanent contacts).
    report
        Pairwise + per-position contact statistics.
    library
        The rotamer library used. Reused across multiple calls to avoid
        re-parsing the (large) EBL.out file.
    pdb_path
        Path to the analyzed structure. When ``analyze()`` was called with a
        pre-parsed :class:`gemmi.Structure`, this falls back to its ``.name``
        (or ``"<gemmi.Structure>"`` if it has none).
    """

    positions: list[PositionRotamers]
    report: ContactReport
    library: RotamerLibrary
    pdb_path: Path

    def positions_dataframe(self) -> pd.DataFrame:  # noqa: F821
        """Return a :class:`pandas.DataFrame` of per-residue scores.

        One row per position. Columns: ``chain``, ``resnum``, ``icode``,
        ``resname``, ``sumcond`` (sum contact degree), ``crwdnes``,
        ``freedom``, ``n_rotamers`` (placed and survived), ``n_rotamers_placed``
        (before pruning), ``fraction_pruned``, ``in_focus``.
        """
        import pandas as pd

        rep = self.report
        return pd.DataFrame(
            {
                "chain": [p.position.chain for p in self.positions],
                "resnum": [p.position.resnum for p in self.positions],
                "icode": [p.position.icode for p in self.positions],
                "resname": [p.position.resname for p in self.positions],
                "sumcond": rep.sum_contact_degree,
                "crwdnes": rep.crwdnes,
                "freedom": rep.freedom,
                "n_rotamers": [len(p.rotamers) for p in self.positions],
                "n_rotamers_placed": [p.num_rotamers_placed for p in self.positions],
                "fraction_pruned": [p.fraction_pruned for p in self.positions],
                "in_focus": [p.in_focus for p in self.positions],
            }
        )

    def contacts_dataframe(self) -> pd.DataFrame:  # noqa: F821
        """Return a :class:`pandas.DataFrame` of pairwise contacts.

        One row per contact pair (the lower-triangular form — each pair
        appears once). Columns: ``chain_i``/``resnum_i``/``icode_i``/
        ``resname_i``, same for ``_j``, and ``degree``.
        """
        import pandas as pd

        pos = [p.position for p in self.positions]
        rows = []
        for c in self.report.contacts:
            pi, pj = pos[c.pos_i], pos[c.pos_j]
            rows.append(
                (
                    pi.chain, pi.resnum, pi.icode, pi.resname,
                    pj.chain, pj.resnum, pj.icode, pj.resname,
                    c.degree,
                )
            )
        return pd.DataFrame(
            rows,
            columns=[
                "chain_i", "resnum_i", "icode_i", "resname_i",
                "chain_j", "resnum_j", "icode_j", "resname_j",
                "degree",
            ],
        )


#: Process-wide cache for the default (auto-downloaded) rotamer library. Lazily
#: populated on the first :func:`analyze` call that doesn't supply its own
#: ``rotamer_library`` — reused by every subsequent call. Parsing ``EBL.out``
#: takes ~5 s on a typical machine and dwarfs the actual analysis on small
#: structures, so the savings here are dramatic. To force a reload (e.g. in
#: tests or after switching libraries), set this back to ``None`` via
#: ``pyconfind.api.DEFAULT_ROTAMER_LIBRARY = None``.
DEFAULT_ROTAMER_LIBRARY: RotamerLibrary | None = None


def _ensure_default_library() -> RotamerLibrary:
    global DEFAULT_ROTAMER_LIBRARY
    if DEFAULT_ROTAMER_LIBRARY is None:
        DEFAULT_ROTAMER_LIBRARY = load_library(cached_rotamer_library())
    return DEFAULT_ROTAMER_LIBRARY


def analyze(
    structure: str | Path | Any,
    rotamer_library: str | Path | RotamerLibrary | None = None,
    *,
    pre_select: str | None = None,
    focus: str | None = None,
    contact_distance: float = 3.0,
    clash_distance: float = 2.0,
    dcut: float = 25.0,
    do_not_count_cb: bool = True,
    renumber: bool = False,
    native_only: bool = False,
    backend: str = "auto",
    assembly: int | str | None = 1,
) -> Analysis:
    """Run the full confind pipeline on a structure.

    Parameters
    ----------
    structure
        Path to a PDB/mmCIF file *or* a pre-parsed :class:`gemmi.Structure`.
        Pass a gemmi structure when you've already loaded (or programmatically
        built) it and don't want pyconfind to re-read the file.
    rotamer_library
        A backbone-dependent rotamer library directory (containing ``EBL.out``
        + ``BEBL.out``) or a pre-loaded :class:`RotamerLibrary`. If ``None``
        (the default), the Dunbrack 2010 library is downloaded once and cached
        per-user (see :func:`pyconfind.cached_rotamer_library`); the parsed
        library is then memoized in :data:`DEFAULT_ROTAMER_LIBRARY` so every
        subsequent call in the same process skips the ~5 s re-parse. For
        batches with a custom library, build a :class:`RotamerLibrary` once
        via :func:`load_library` and pass it explicitly.
    pre_select
        MSL selection string (the ``--psel`` flag). Only residues whose CA
        atom satisfies this selection are kept in the structure before
        anything else runs.
    focus
        MSL selection string (the ``--sel`` flag). Rotamers are only placed
        and contacts only computed for residues whose CA atom satisfies this
        selection; the rest of the structure is still present for backbone
        collision detection but is excluded from the output.
    contact_distance
        Sidechain-sidechain contact cutoff in Å. C++ default 3.0.
    clash_distance
        Backbone-clash cutoff used while pruning rotamers. C++ default 2.0.
    dcut
        Pair-cutoff in Å (CA-CA) below which a pair is considered for the
        contact-degree computation. C++ default 25.0.
    do_not_count_cb
        If ``True`` (the C++ default), CB is excluded from backbone clash
        checks for non-ALA residues.
    renumber
        If ``True``, renumber residues per chain starting at 1 (the ``--ren``
        flag in C++ confind).
    native_only
        New mode: only substitute the *native* AA at each position instead
        of all 18. Still uses all rotamers of that AA.
    backend
        Contact-degree backend: ``"numba"`` (JIT-compiled, multi-threaded),
        ``"python"`` (pure NumPy/SciPy reference), or ``"auto"`` (the default —
        use Numba if importable, else fall back to the reference). Both
        backends produce results identical to ~1e-6 (the output precision).
    assembly
        Biological assembly to analyze. ``1`` (the default) picks the first
        bio assembly declared in the file — important for structures whose
        asymmetric unit contains multiple independent copies (e.g. 5TRU has
        two Fab/CTLA-4 complexes; ``assembly=1`` analyzes just one). Pass
        ``None`` to use the asymmetric unit as-is.
    """
    if isinstance(rotamer_library, RotamerLibrary):
        library = rotamer_library
    elif rotamer_library is None:
        library = _ensure_default_library()
    else:
        library = load_library(rotamer_library)

    if isinstance(structure, str | Path):
        pdb_path = Path(structure)
        # gemmi reads both PDB and mmCIF; format is auto-detected.
        atoms = read_structure(pdb_path, renumber=renumber, assembly=assembly)
    else:
        # Assume an already-parsed gemmi.Structure (duck-typed).
        atoms = atoms_from_gemmi_structure(
            structure, renumber=renumber, assembly=assembly,
        )
        pdb_path = Path(getattr(structure, "name", "") or "<gemmi.Structure>")

    # --psel: keep only residues whose CA satisfies the pre-selection. The
    # C++ applies this on the raw structure before protein-only filtering;
    # we apply it after reading (which already drops non-protein atoms).
    if pre_select is not None:
        pos_mask = select_residue_mask(atoms, pre_select)
        atoms = filter_atoms_by_position(atoms, pos_mask)

    positions = positions_from_atoms(atoms)

    focus_mask = None
    if focus is not None:
        focus_mask = select_residue_mask(atoms, focus)

    rot_results = build_position_rotamers(
        positions,
        library,
        clash_dist=clash_distance,
        do_not_count_cb=do_not_count_cb,
        native_only=native_only,
        focus_mask=focus_mask,
    )
    contact_fn = _select_contact_backend(backend)
    report = contact_fn(
        rot_results,
        dcut=dcut,
        contact_dist=contact_distance,
    )
    return Analysis(
        positions=rot_results,
        report=report,
        library=library,
        pdb_path=pdb_path,
    )
