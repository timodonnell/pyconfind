"""Structure parsing on real PDB/mmCIF entries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind.pdb import LEGAL_RESIDUE_NAMES, position_iter, read_pdb, read_structure
from tests.conftest import REAL_STRUCTURES


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_reads_real_structure(structures_dir: Path, name: str) -> None:
    atoms = read_structure(structures_dir / f"{name}.pdb")
    assert len(atoms) > 0
    assert atoms.num_positions > 0
    # Only legal protein residues survive the default filter.
    assert set(atoms.resname.tolist()) <= LEGAL_RESIDUE_NAMES
    # Every position's atoms are contiguous.
    for s in position_iter(atoms):
        pis = atoms.position_index[s]
        assert (pis == pis[0]).all()


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_gemmi_matches_hand_parser(structures_dir: Path, name: str) -> None:
    """The gemmi reader and the dependency-free hand PDB parser must agree."""
    g = read_structure(structures_dir / f"{name}.pdb")
    h = read_pdb(structures_dir / f"{name}.pdb")
    assert list(g.chain) == list(h.chain)
    assert list(g.resnum) == list(h.resnum)
    assert list(g.icode) == list(h.icode)
    assert list(g.resname) == list(h.resname)
    assert list(g.name) == list(h.name)
    assert list(g.position_index) == list(h.position_index)
    np.testing.assert_allclose(g.xyz, h.xyz, atol=1e-6)


def test_altloc_filtered(structures_dir: Path) -> None:
    """1EJG is an ultra-high-res structure with alternate locations; only the
    primary (A/blank) conformer is kept by default."""
    atoms = read_structure(structures_dir / "1EJG.pdb")
    assert set(atoms.altloc.tolist()) <= {"", "A"}
    # Each (position, atom name) appears at most once after altloc filtering.
    seen = set()
    for i in range(len(atoms)):
        key = (int(atoms.position_index[i]), str(atoms.name[i]))
        assert key not in seen, f"duplicate atom {key} survived altloc filter"
        seen.add(key)


def test_renumber(structures_dir: Path) -> None:
    plain = read_structure(structures_dir / "1CRN.pdb")
    renum = read_structure(structures_dir / "1CRN.pdb", renumber=True)
    assert renum.num_positions == plain.num_positions
    first_chain = renum.chain[0]
    chain_resnums = sorted(set(renum.resnum[renum.chain == first_chain].tolist()))
    assert chain_resnums == list(range(1, len(chain_resnums) + 1))
    assert set(renum.icode.tolist()) == {""}


def test_empty_input(tmp_path: Path) -> None:
    """A file with no ATOM records yields an empty structure, not an error."""
    p = tmp_path / "empty.pdb"
    p.write_text("HEADER    NOTHING HERE\nEND\n")
    atoms = read_pdb(p)
    assert len(atoms) == 0
    assert atoms.num_positions == 0
    assert position_iter(atoms) == []
