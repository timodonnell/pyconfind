"""High-level library API.

For library use (notebooks, pipelines, scripts), call :func:`analyze`. It
takes a PDB path and returns the :class:`Analysis` result with all the
pieces — surviving rotamers per position, contact-degree report, and the
loaded library — so downstream code can format, slice, or compute further
without re-loading anything.

Example
-------
>>> from pyconfind import analyze
>>> result = analyze("input.pdb", rotamer_library="path/to/rotlibs")
>>> for c in result.report.contacts:
...     print(c.pos_i, c.pos_j, c.degree)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .build import PositionRotamers, build_position_rotamers
from .contacts import ContactReport, compute_contacts
from .pdb import filter_atoms_by_position, read_pdb
from .rotlib import RotamerLibrary, load_library
from .selection import select_residue_mask
from .structure import positions_from_atoms


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
        The PDB that was analyzed.
    """

    positions: list[PositionRotamers]
    report: ContactReport
    library: RotamerLibrary
    pdb_path: Path


def analyze(
    pdb_path: str | Path,
    rotamer_library: str | Path | RotamerLibrary,
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
) -> Analysis:
    """Run the full confind pipeline on a PDB file.

    Parameters
    ----------
    pdb_path
        Path to the input PDB.
    rotamer_library
        Either a path to a rotamer library directory (e.g. ``./rotlibs``
        containing ``EBL.out`` + ``BEBL.out``) or a single ``EBL.out`` file
        for bb-indep operation, or a pre-loaded :class:`RotamerLibrary`.
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
    """
    pdb_path = Path(pdb_path)
    if isinstance(rotamer_library, RotamerLibrary):
        library = rotamer_library
    else:
        library = load_library(rotamer_library)
    atoms = read_pdb(pdb_path, renumber=renumber)

    # --psel: keep only residues whose CA satisfies the pre-selection. The
    # C++ applies this on the raw structure before protein-only filtering;
    # we apply it after read_pdb (which already drops non-protein atoms).
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
