"""Rotamer building + per-pair/per-position scores, on real structures.

Each row type (crwdnes, contact, sumcond, freedom) must match the C++ golden
to <1e-5. Runs locally with the full library; self-skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.structure import dihedral_deg
from tests.conftest import REAL_STRUCTURES


def test_dihedral_sign_convention() -> None:
    """IUPAC convention: positive when D rotates clockwise viewed from B->C."""
    a, b, c, d = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
    )
    assert abs(dihedral_deg(a, b, c, d) - (-90.0)) < 1e-9


def _golden_rows(text: str, tag: str) -> dict:
    out = {}
    for line in text.splitlines():
        p = line.split("\t")
        if p[0] == tag:
            if tag == "contact":
                out[(p[1], p[2])] = float(p[3])
            else:
                out[p[1]] = float(p[2]) if p[2] not in ("-nan", "nan") else np.nan
    return out


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_per_position_and_pair_scores_match_cpp(
    structures_dir: Path, golden_dir: Path, rotlib_dir: Path, name: str
) -> None:
    golden_path = golden_dir / f"{name}.cont"
    if not golden_path.exists():
        pytest.skip("golden missing")
    a = analyze(structures_dir / f"{name}.pdb", rotamer_library=rotlib_dir)
    mine = format_confind_text(a.positions, a.report)
    gold = golden_path.read_text()
    for tag in ("crwdnes", "sumcond", "freedom"):
        g = _golden_rows(gold, tag)
        m = _golden_rows(mine, tag)
        assert set(g) == set(m), f"{name} {tag} position sets differ"
        for k in g:
            if np.isnan(g[k]):
                assert np.isnan(m[k]), f"{name} {tag} {k}: expected NaN"
            else:
                assert abs(g[k] - m[k]) < 1e-5, f"{name} {tag} {k}: {m[k]} vs {g[k]}"
    gc = _golden_rows(gold, "contact")
    mc = _golden_rows(mine, "contact")
    assert set(gc) == set(mc), f"{name} contact pair set differs"
    for k in gc:
        assert abs(gc[k] - mc[k]) < 1e-5, f"{name} contact {k}: {mc[k]} vs {gc[k]}"


def test_native_only_handles_gly_pro(structures_dir: Path, rotlib_dir: Path) -> None:
    """--native-only must not crash on GLY (no rotamers) and must place only
    native-AA rotamers (regression for the GLY KeyError)."""
    a = analyze(structures_dir / "1UBQ.pdb", rotamer_library=rotlib_dir, native_only=True)
    glys = [p for p in a.positions if p.position.resname == "GLY"]
    pros = [p for p in a.positions if p.position.resname == "PRO"]
    assert glys and all(len(p.rotamers) == 0 for p in glys)  # GLY has no rotamers
    assert pros and all(len(p.rotamers) > 0 for p in pros)   # PRO does
    # Every surviving rotamer is the native AA.
    for p in a.positions:
        for rot in p.rotamers:
            assert rot.aa == p.position.resname
