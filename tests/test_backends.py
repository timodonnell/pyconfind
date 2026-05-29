"""The Numba and pure-Python contact backends must agree, on real structures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.build import build_position_rotamers
from pyconfind.contacts import compute_contacts
from pyconfind.pdb import read_structure
from pyconfind.structure import positions_from_atoms
from tests.conftest import REAL_STRUCTURES

numba_backend = pytest.importorskip("pyconfind.contacts_numba")


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_numba_matches_python(structures_dir: Path, mini_rotlib: Path, name: str) -> None:
    """Backends agree to well within print precision, using the CI mini library."""
    from pyconfind.rotlib import load_library

    atoms = read_structure(structures_dir / f"{name}.pdb")
    positions = positions_from_atoms(atoms)
    lib = load_library(mini_rotlib)
    res = build_position_rotamers(positions, lib)

    rep_py = compute_contacts(res)
    rep_nb = numba_backend.compute_contacts_numba(res)

    py = {(c.pos_i, c.pos_j): c.degree for c in rep_py.contacts}
    nb = {(c.pos_i, c.pos_j): c.degree for c in rep_nb.contacts}
    assert set(py) == set(nb), f"contact sets differ for {name}"
    for k in py:
        assert abs(py[k] - nb[k]) < 1e-9, f"{name} {k}"
    np.testing.assert_allclose(rep_py.sum_contact_degree, rep_nb.sum_contact_degree, atol=1e-9)
    np.testing.assert_allclose(rep_py.freedom, rep_nb.freedom, atol=1e-9, equal_nan=True)
    assert format_confind_text(res, rep_py) == format_confind_text(res, rep_nb)


def test_backend_selection() -> None:
    from pyconfind.api import _select_contact_backend

    assert _select_contact_backend("auto") is numba_backend.compute_contacts_numba
    assert _select_contact_backend("python") is compute_contacts


def test_analyze_backend_equivalence(structures_dir: Path, mini_rotlib: Path) -> None:
    a_py = analyze(structures_dir / "1UBQ.pdb", rotamer_library=mini_rotlib, backend="python")
    a_nb = analyze(structures_dir / "1UBQ.pdb", rotamer_library=mini_rotlib, backend="numba")
    assert format_confind_text(a_py.positions, a_py.report) == format_confind_text(
        a_nb.positions, a_nb.report
    )
