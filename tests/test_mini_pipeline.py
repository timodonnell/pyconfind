"""End-to-end regression on real structures with the bundled mini library.

Runs anywhere (no production library needed), so it executes in CI and guards
the full place -> prune -> contact -> output pipeline. The numbers are a
snapshot of pyconfind's own output on the mini (truncated) library — not the
C++ reference values (those need the full library and are checked in the other
modules).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyconfind import analyze, format_confind_text
from tests.conftest import REAL_STRUCTURES

SNAPSHOTS = Path(__file__).resolve().parent / "data" / "snapshots"


@pytest.mark.parametrize("name", REAL_STRUCTURES)
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_mini_pipeline_matches_snapshot(
    structures_dir: Path, mini_rotlib: Path, name: str, backend: str
) -> None:
    if backend == "numba":
        pytest.importorskip("numba")
    snap = SNAPSHOTS / f"{name}.mini.cont"
    a = analyze(structures_dir / f"{name}.pdb", rotamer_library=mini_rotlib, backend=backend)
    rendered = format_confind_text(a.positions, a.report)
    assert rendered == snap.read_text(), f"{name} ({backend}) regressed vs snapshot"


def test_mini_library_loads(mini_rotlib: Path) -> None:
    from pyconfind import load_library

    lib = load_library(mini_rotlib)
    assert lib.is_backbone_dependent
    confs, weights = lib.rotamers_for("ARG", phi=None, psi=None)
    assert confs.shape[0] == weights.size > 0
