"""Tests for the PDB loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyconfind.pdb import LEGAL_RESIDUE_NAMES, position_iter, read_pdb


def test_read_example0000(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0000.pdb")
    # The PDB has 27 ATOM records, all legal.
    assert len(atoms) == 27
    # 3 positions: chain A, residues 1, 2, 3 — but residue 2 has TWO identities
    # (ILE and LEU) so there are still 3 positions.
    assert atoms.num_positions == 3
    # First atom is ALA N at (2.143, 1.328, 0.000)
    np.testing.assert_allclose(atoms.xyz[0], [2.143, 1.328, 0.000])
    assert atoms.chain[0] == "A"
    assert atoms.resnum[0] == 1
    assert atoms.resname[0] == "ALA"
    assert atoms.name[0] == "N"


def test_read_example0000_multi_identity(examples_dir: Path) -> None:
    """Position 2 has both ILE and LEU identities."""
    atoms = read_pdb(examples_dir / "example0000.pdb")
    # Find atoms in position 1 (0-indexed, so residue 2)
    in_pos1 = atoms.position_index == 1
    resnames_in_pos1 = set(atoms.resname[in_pos1].tolist())
    assert resnames_in_pos1 == {"ILE", "LEU"}
    # Identity index distinguishes them
    ile_mask = in_pos1 & (atoms.resname == "ILE")
    leu_mask = in_pos1 & (atoms.resname == "LEU")
    assert (atoms.identity_index[ile_mask] == 0).all()
    assert (atoms.identity_index[leu_mask] == 1).all()


def test_read_legal_only_filters(tmp_path: Path) -> None:
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    2  O   HOH A 100       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      3  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    )
    p = tmp_path / "t.pdb"
    p.write_text(pdb)
    atoms = read_pdb(p)
    # HOH is dropped
    assert len(atoms) == 2
    assert "HOH" not in atoms.resname.tolist()
    # If we disable filtering, all 3 atoms come through
    atoms_all = read_pdb(p, legal_only=False)
    assert len(atoms_all) == 3


def test_altloc_filter(tmp_path: Path) -> None:
    pdb = (
        "ATOM      1  N  AALA A   1       0.000   0.000   0.000  0.60  0.00           N\n"
        "ATOM      2  N  BALA A   1       0.000   0.000   0.000  0.40  0.00           N\n"
        "ATOM      3  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    )
    p = tmp_path / "t.pdb"
    p.write_text(pdb)
    atoms = read_pdb(p, altloc="A")
    # Keep altloc A and blank, drop B.
    assert len(atoms) == 2
    assert "B" not in atoms.altloc.tolist()


def test_multi_model_only_first(tmp_path: Path) -> None:
    pdb = (
        "MODEL        1\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      2  N   ALA A   1      10.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\n"
    )
    p = tmp_path / "t.pdb"
    p.write_text(pdb)
    atoms = read_pdb(p)
    assert len(atoms) == 1
    np.testing.assert_allclose(atoms.xyz[0], [0.0, 0.0, 0.0])


def test_position_iter_groups_residues(examples_dir: Path) -> None:
    atoms = read_pdb(examples_dir / "example0000.pdb")
    slices = position_iter(atoms)
    assert len(slices) == atoms.num_positions
    # All atoms in each slice share the same position index.
    for s in slices:
        pis = atoms.position_index[s]
        assert (pis == pis[0]).all()


def test_renumber(tmp_path: Path) -> None:
    pdb = (
        "ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A  10       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  N   ALA A  20       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      4  N   ALA B   5       0.000   0.000   0.000  1.00  0.00           N\n"
    )
    p = tmp_path / "t.pdb"
    p.write_text(pdb)
    atoms = read_pdb(p, renumber=True)
    # Chain A residues -> 1, 2; chain B residue -> 1
    assert list(atoms.resnum) == [1, 1, 2, 1]


def test_legal_residue_names_includes_his_variants() -> None:
    for n in ("HIS", "HSD", "HSE", "HSC", "HSP"):
        assert n in LEGAL_RESIDUE_NAMES


def test_empty_pdb(tmp_path: Path) -> None:
    p = tmp_path / "empty.pdb"
    p.write_text("")
    atoms = read_pdb(p)
    assert len(atoms) == 0
    assert atoms.num_positions == 0
    assert position_iter(atoms) == []
