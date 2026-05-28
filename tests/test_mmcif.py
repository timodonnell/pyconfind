"""mmCIF support: reading .cif must match reading the equivalent .pdb.

Structure parsing goes through gemmi, which reads both PDB and mmCIF. These
tests confirm the two formats yield identical atoms and identical pipeline
output, so all the PDB-based validation (incl. byte-identity vs the C++
reference) carries over to mmCIF inputs. They run in CI on the bundled
fixtures (no production rotamer library needed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.pdb import read_structure

DATA = Path(__file__).resolve().parent / "data"
EXAMPLES = DATA / "examples"
MINI_ROTLIB = DATA / "mini_rotlib"

_CASES = ["example0000", "example0002", "example0007"]


@pytest.mark.parametrize("name", _CASES)
def test_cif_atoms_match_pdb(name: str) -> None:
    pdb = read_structure(EXAMPLES / f"{name}.pdb")
    cif = read_structure(EXAMPLES / f"{name}.cif")
    assert list(pdb.chain) == list(cif.chain)
    assert list(pdb.resnum) == list(cif.resnum)
    assert list(pdb.icode) == list(cif.icode)
    assert list(pdb.resname) == list(cif.resname)
    assert list(pdb.name) == list(cif.name)
    assert list(pdb.position_index) == list(cif.position_index)
    np.testing.assert_allclose(pdb.xyz, cif.xyz, atol=1e-6)


@pytest.mark.parametrize("name", _CASES)
def test_cif_pipeline_matches_pdb(name: str) -> None:
    """Full analyze() output must be identical for .cif and .pdb inputs."""
    a_pdb = analyze(EXAMPLES / f"{name}.pdb", rotamer_library=MINI_ROTLIB)
    a_cif = analyze(EXAMPLES / f"{name}.cif", rotamer_library=MINI_ROTLIB)
    assert format_confind_text(a_pdb.positions, a_pdb.report) == format_confind_text(
        a_cif.positions, a_cif.report
    )


def test_cif_byte_identical_to_cpp_golden(rotlib_dir: Path) -> None:
    """With the full library, the .cif path reproduces the C++ golden output."""
    golden = Path(__file__).resolve().parent / "golden" / "example0002.cont"
    if not golden.exists():
        pytest.skip("golden output missing")
    a = analyze(EXAMPLES / "example0002.cif", rotamer_library=rotlib_dir)
    assert format_confind_text(a.positions, a.report) == golden.read_text()
