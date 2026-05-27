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

from dataclasses import dataclass
from pathlib import Path

from .build import PositionRotamers, build_position_rotamers
from .contacts import ContactReport, compute_contacts
from .pdb import read_pdb
from .rotlib import RotamerLibrary, load_library
from .structure import positions_from_atoms


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
    pre_select: str | None = None,  # not yet implemented
    focus: str | None = None,        # not yet implemented
    contact_distance: float = 3.0,
    clash_distance: float = 2.0,
    dcut: float = 25.0,
    do_not_count_cb: bool = True,
    renumber: bool = False,
    native_only: bool = False,
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
    pre_select, focus
        Reserved for the ``--psel`` / ``--sel`` C++ flags. Not implemented
        yet — passing a non-None value will raise :class:`NotImplementedError`.
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
    """
    if pre_select is not None or focus is not None:
        raise NotImplementedError(
            "--psel / --sel selection is not yet implemented in pyconfind."
        )
    pdb_path = Path(pdb_path)
    if isinstance(rotamer_library, RotamerLibrary):
        library = rotamer_library
    else:
        library = load_library(rotamer_library)
    atoms = read_pdb(pdb_path, renumber=renumber)
    positions = positions_from_atoms(atoms)
    rot_results = build_position_rotamers(
        positions,
        library,
        clash_dist=clash_distance,
        do_not_count_cb=do_not_count_cb,
        native_only=native_only,
    )
    report = compute_contacts(
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
