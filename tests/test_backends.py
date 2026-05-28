"""The Numba and pure-Python contact backends must agree."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind.build import build_position_rotamers
from pyconfind.contacts import compute_contacts
from pyconfind.output import format_confind_text
from pyconfind.pdb import read_pdb
from pyconfind.rotlib import load_library
from pyconfind.structure import positions_from_atoms

numba_backend = pytest.importorskip("pyconfind.contacts_numba")


@pytest.mark.parametrize(
    "pdb_name",
    ["example0000.pdb", "example0002.pdb", "example0007.pdb", "example0008.pdb"],
)
def test_numba_matches_python(examples_dir: Path, rotlib_dir: Path, pdb_name: str) -> None:
    atoms = read_pdb(examples_dir / pdb_name)
    if len(atoms) == 0:
        pytest.skip(f"{pdb_name} empty")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    res = build_position_rotamers(positions, lib)

    rep_py = compute_contacts(res)
    rep_nb = numba_backend.compute_contacts_numba(res)

    # Same set of contacts, degrees agree to well within print precision.
    py = {(c.pos_i, c.pos_j): c.degree for c in rep_py.contacts}
    nb = {(c.pos_i, c.pos_j): c.degree for c in rep_nb.contacts}
    assert set(py) == set(nb), f"contact sets differ for {pdb_name}"
    for k in py:
        assert abs(py[k] - nb[k]) < 1e-9, f"{pdb_name} {k}: {py[k]} vs {nb[k]}"

    np.testing.assert_allclose(
        rep_py.sum_contact_degree, rep_nb.sum_contact_degree, atol=1e-9
    )
    np.testing.assert_allclose(rep_py.freedom, rep_nb.freedom, atol=1e-9, equal_nan=True)

    # And the rendered text output is byte-identical between backends.
    assert format_confind_text(res, rep_py) == format_confind_text(res, rep_nb)


def test_backend_selection_auto_prefers_numba(examples_dir: Path, rotlib_dir: Path) -> None:
    from pyconfind.api import _select_contact_backend

    auto = _select_contact_backend("auto")
    assert auto is numba_backend.compute_contacts_numba
    assert _select_contact_backend("python") is compute_contacts


def test_analyze_backend_equivalence(examples_dir: Path, rotlib_dir: Path) -> None:
    from pyconfind import analyze

    lib = load_library(rotlib_dir)
    a_py = analyze(examples_dir / "example0002.pdb", rotamer_library=lib, backend="python")
    a_nb = analyze(examples_dir / "example0002.pdb", rotamer_library=lib, backend="numba")
    assert format_confind_text(a_py.positions, a_py.report) == format_confind_text(
        a_nb.positions, a_nb.report
    )
