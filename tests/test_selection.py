"""Atom-selection language + --psel/--sel, validated against C++ on 1UBQ.

1UBQ is single-chain, so the boolean grammar (AND/OR/NOT/parens, WITHIN) is
exercised via resi/resn/name predicates. Byte-identity vs the C++ binary is
the correctness gate (skips when the binary/library are absent, e.g. in CI).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.pdb import read_structure
from pyconfind.rotlib import load_library
from pyconfind.selection import select_atom_mask, select_residue_mask
from pyconfind.structure import positions_from_atoms

_CPP = (
    Path(__file__).resolve().parents[1]
    / "original-source" / "confind-msl" / "mslib" / "bin" / "confind"
)


def test_select_atom_mask_basic(structures_dir: Path) -> None:
    atoms = read_structure(structures_dir / "1UBQ.pdb")
    chain_a = select_atom_mask(atoms, "chain A")
    assert chain_a.all()  # 1UBQ is all chain A
    ca = select_atom_mask(atoms, "name CA")
    assert set(atoms.name[ca].tolist()) == {"CA"}
    ca_resn = select_atom_mask(atoms, "name CA and resn GLY")
    assert set(atoms.resname[ca_resn].tolist()) == {"GLY"}


def test_select_resi_range_and_list(structures_dir: Path) -> None:
    atoms = read_structure(structures_dir / "1UBQ.pdb")
    rng = select_atom_mask(atoms, "resi 2-5")
    assert atoms.resnum[rng].min() >= 2 and atoms.resnum[rng].max() <= 5
    lst = select_atom_mask(atoms, "resi 2+4+6")
    assert set(atoms.resnum[lst].tolist()) == {2, 4, 6}


def test_select_precedence_and_not(structures_dir: Path) -> None:
    atoms = read_structure(structures_dir / "1UBQ.pdb")
    positions = positions_from_atoms(atoms)
    # AND binds tighter than OR: resi1-10 ∪ (resi 20-40 ∧ GLY)
    m = select_residue_mask(atoms, "resi 1-10 OR resi 20-40 AND resn GLY")
    got = {positions[i].resnum for i in np.flatnonzero(m)}
    exp = {p.resnum for p in positions if 1 <= p.resnum <= 10}
    exp |= {p.resnum for p in positions if 20 <= p.resnum <= 40 and p.resname == "GLY"}
    assert got == exp
    # Parens override precedence
    m2 = select_residue_mask(atoms, "(resi 1-10 OR resi 20-40) AND NOT resn GLY")
    got2 = {positions[i].resnum for i in np.flatnonzero(m2)}
    exp2 = {p.resnum for p in positions
            if (1 <= p.resnum <= 10 or 20 <= p.resnum <= 40) and p.resname != "GLY"}
    assert got2 == exp2


def test_select_within(structures_dir: Path) -> None:
    atoms = read_structure(structures_dir / "1UBQ.pdb")
    near = select_residue_mask(atoms, "NAME CA WITHIN 8 OF resi 40")
    assert 0 < near.sum() < atoms.num_positions


@pytest.mark.parametrize(
    ("flag", "sel", "kw"),
    [
        ("--sel", "resi 1-30", {"focus": "resi 1-30"}),
        ("--sel", "resn GLY+ALA", {"focus": "resn GLY+ALA"}),
        ("--sel", "NAME CA WITHIN 8 OF resi 40", {"focus": "NAME CA WITHIN 8 OF resi 40"}),
        ("--sel", "resi 1-40 AND NOT resi 15-25", {"focus": "resi 1-40 AND NOT resi 15-25"}),
        ("--psel", "resi 1-50", {"pre_select": "resi 1-50"}),
        ("--psel", "resi 1-20 OR resi 60-76", {"pre_select": "resi 1-20 OR resi 60-76"}),
    ],
)
def test_sel_psel_byte_identical(
    structures_dir: Path, rotlib_dir: Path, flag: str, sel: str, kw: dict
) -> None:
    if not _CPP.exists():
        pytest.skip("C++ reference binary not built")
    pdb = structures_dir / "1UBQ.pdb"
    lib = load_library(rotlib_dir)
    out = Path(tempfile.mktemp(suffix=".cont"))
    subprocess.run(
        [str(_CPP), "--p", str(pdb), "--rLib", str(rotlib_dir), flag, sel, "--o", str(out)],
        check=True, capture_output=True,
    )
    cpp = out.read_text()
    out.unlink()
    a = analyze(pdb, rotamer_library=lib, **kw)
    assert format_confind_text(a.positions, a.report) == cpp, f"{flag} '{sel}' differs"


def test_cli_sel_flag(structures_dir: Path, mini_rotlib: Path, tmp_path: Path) -> None:
    """The --sel CLI flag restricts output to the focus residues (CI: mini lib)."""
    out = tmp_path / "out.cont"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(structures_dir / "1UBQ.pdb"),
            "--rLib", str(mini_rotlib),
            "--o", str(out),
            "--sel", "resi 1-10",
        ],
        check=True, capture_output=True,
    )
    resnums = {
        int(line.split("\t")[1].split(",")[1])
        for line in out.read_text().splitlines()
        if line.startswith("sumcond")
    }
    assert resnums == set(range(1, 11))
