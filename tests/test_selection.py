"""Tests for the atom-selection language and --psel/--sel integration."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.pdb import read_pdb
from pyconfind.rotlib import load_library
from pyconfind.selection import select_atom_mask, select_residue_mask
from pyconfind.structure import positions_from_atoms


def test_select_atom_mask_basic(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0002.pdb")
    # CHAIN
    chain_a = select_atom_mask(atoms, "chain A")
    assert chain_a.sum() > 0
    assert set(atoms.chain[chain_a].tolist()) == {"A"}
    # NAME
    ca = select_atom_mask(atoms, "name CA")
    assert set(atoms.name[ca].tolist()) == {"CA"}
    # combined
    ca_a = select_atom_mask(atoms, "name CA and chain A")
    assert (ca_a == (chain_a & ca)).all()


def test_select_resi_range_and_list(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0002.pdb")
    rng = select_atom_mask(atoms, "resi 2-5")
    assert atoms.resnum[rng].min() >= 2 and atoms.resnum[rng].max() <= 5
    lst = select_atom_mask(atoms, "resi 2+4+6")
    assert set(atoms.resnum[lst].tolist()) == {2, 4, 6}


def test_select_precedence_and_not(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0002.pdb")
    # AND binds tighter than OR
    m = select_residue_mask(atoms, "resi 2 OR resi 3 AND chain B")
    positions = positions_from_atoms(atoms)
    got = {(positions[i].chain, positions[i].resnum) for i in np.flatnonzero(m)}
    # resi2 (all chains) ∪ (resi3 ∧ chainB)
    expected_resi2 = {(p.chain, p.resnum) for p in positions if p.resnum == 2}
    expected_b3 = {(p.chain, p.resnum) for p in positions if p.resnum == 3 and p.chain == "B"}
    assert got == expected_resi2 | expected_b3
    # parens override
    m2 = select_residue_mask(atoms, "(resi 2 OR resi 3) AND chain B")
    got2 = {(positions[i].chain, positions[i].resnum) for i in np.flatnonzero(m2)}
    assert got2 == {("B", 2), ("B", 3)}


def test_select_within(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0002.pdb")
    near = select_residue_mask(atoms, "NAME CA WITHIN 8 OF CHAIN B")
    assert near.sum() > 0


# --- byte-identity vs C++ for --sel / --psel ------------------------------

_CPP = (
    Path(__file__).resolve().parents[1]
    / "original-source" / "confind-msl" / "mslib" / "bin" / "confind"
)


def _cpp_output(pdb: Path, rotlib: Path, flag: str, sel: str) -> str:
    out = Path(tempfile.mktemp(suffix=".cont"))
    subprocess.run(
        [str(_CPP), "--p", str(pdb), "--rLib", str(rotlib), flag, sel, "--o", str(out)],
        check=True, capture_output=True,
    )
    txt = out.read_text()
    out.unlink()
    return txt


@pytest.mark.parametrize(
    ("flag", "sel", "kw"),
    [
        ("--sel", "chain A", {"focus": "chain A"}),
        ("--sel", "resi 2-5", {"focus": "resi 2-5"}),
        ("--sel", "NAME CA WITHIN 8 OF CHAIN B", {"focus": "NAME CA WITHIN 8 OF CHAIN B"}),
        ("--sel", "resi 2 AND NOT chain A", {"focus": "resi 2 AND NOT chain A"}),
        ("--psel", "chain B", {"pre_select": "chain B"}),
        ("--psel", "chain A OR chain C", {"pre_select": "chain A OR chain C"}),
        ("--psel", "resi 1-6", {"pre_select": "resi 1-6"}),
    ],
)
def test_sel_psel_byte_identical(
    examples_dir: Path, rotlib_dir: Path, flag: str, sel: str, kw: dict
) -> None:
    if not _CPP.exists():
        pytest.skip("C++ reference binary not built")
    pdb = examples_dir / "example0002.pdb"
    lib = load_library(rotlib_dir)
    cpp = _cpp_output(pdb, rotlib_dir, flag, sel)
    a = analyze(pdb, rotamer_library=lib, **kw)
    mine = format_confind_text(a.positions, a.report)
    assert mine == cpp, f"{flag} '{sel}' differs from C++"


def test_cli_sel_flag(examples_dir: Path, rotlib_dir: Path, tmp_path: Path) -> None:
    """The --sel CLI flag should restrict output to focus residues."""
    out = tmp_path / "out.cont"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(examples_dir / "example0002.pdb"),
            "--rLib", str(rotlib_dir),
            "--o", str(out),
            "--sel", "chain A",
        ],
        check=True, capture_output=True,
    )
    text = out.read_text()
    # Only chain A positions in sumcond rows.
    sumcond_chains = {
        line.split("\t")[1].split(",")[0]
        for line in text.splitlines()
        if line.startswith("sumcond")
    }
    assert sumcond_chains == {"A"}
