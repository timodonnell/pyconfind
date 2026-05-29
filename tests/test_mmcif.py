"""mmCIF support: reading .cif must match the equivalent .pdb (real structures).

Parsing goes through gemmi (both formats). These confirm identical atoms and
identical pipeline output between formats, so the PDB-based byte-identity
validation carries over to mmCIF. Runs in CI on the bundled mini library.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.pdb import read_structure
from tests.conftest import REAL_STRUCTURES


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_cif_atoms_match_pdb(structures_dir: Path, name: str) -> None:
    pdb = read_structure(structures_dir / f"{name}.pdb")
    cif = read_structure(structures_dir / f"{name}.cif")
    assert list(pdb.chain) == list(cif.chain)
    assert list(pdb.resnum) == list(cif.resnum)
    assert list(pdb.icode) == list(cif.icode)
    assert list(pdb.resname) == list(cif.resname)
    assert list(pdb.name) == list(cif.name)
    assert list(pdb.position_index) == list(cif.position_index)
    # Coordinates agree to PDB's 3-decimal precision (mmCIF may carry more).
    np.testing.assert_allclose(pdb.xyz, cif.xyz, atol=1e-3)


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_cif_pipeline_matches_pdb(structures_dir: Path, mini_rotlib: Path, name: str) -> None:
    a_pdb = analyze(structures_dir / f"{name}.pdb", rotamer_library=mini_rotlib)
    a_cif = analyze(structures_dir / f"{name}.cif", rotamer_library=mini_rotlib)
    assert format_confind_text(a_pdb.positions, a_pdb.report) == format_confind_text(
        a_cif.positions, a_cif.report
    )


def test_cif_byte_identical_to_cpp_golden(
    structures_dir: Path, golden_dir: Path, rotlib_dir: Path
) -> None:
    """With the full library, the .cif path reproduces the C++ golden."""
    golden = golden_dir / "1CRN.cont"
    if not golden.exists():
        pytest.skip("golden missing")
    a = analyze(structures_dir / "1CRN.cif", rotamer_library=rotlib_dir)
    assert format_confind_text(a.positions, a.report) == golden.read_text()
