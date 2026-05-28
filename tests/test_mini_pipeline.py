"""End-to-end regression tests on the bundled mini fixture.

These run anywhere (no 100+ MB production library needed), so they execute in
CI and guard the full place -> prune -> contact -> output pipeline against
regressions. The numbers are *not* the C++ reference values (the mini library
is a truncation); they are a snapshot of pyconfind's own output. The C++
byte-identity tests live in the other test modules and run locally where the
full library is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyconfind import analyze, format_confind_text

DATA = Path(__file__).resolve().parent / "data"
MINI_ROTLIB = DATA / "mini_rotlib"
EXAMPLES = DATA / "examples"
SNAPSHOTS = DATA / "snapshots"

_CASES = ["example0000", "example0002", "example0007"]


@pytest.mark.parametrize("name", _CASES)
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_mini_pipeline_matches_snapshot(name: str, backend: str) -> None:
    if backend == "numba":
        pytest.importorskip("numba")
    pdb = EXAMPLES / f"{name}.pdb"
    snap = SNAPSHOTS / f"{name}.mini.cont"
    analysis = analyze(pdb, rotamer_library=MINI_ROTLIB, backend=backend)
    rendered = format_confind_text(analysis.positions, analysis.report)
    assert rendered == snap.read_text(), (
        f"{name} ({backend}) diverged from snapshot — pipeline regression"
    )


def test_backends_agree_on_mini() -> None:
    """Both contact backends must produce identical output on the fixture."""
    pytest.importorskip("numba")
    pdb = EXAMPLES / "example0002.pdb"
    a_py = analyze(pdb, rotamer_library=MINI_ROTLIB, backend="python")
    a_nb = analyze(pdb, rotamer_library=MINI_ROTLIB, backend="numba")
    assert format_confind_text(a_py.positions, a_py.report) == format_confind_text(
        a_nb.positions, a_nb.report
    )


def test_mini_library_loads() -> None:
    from pyconfind import load_library

    lib = load_library(MINI_ROTLIB)
    assert lib.is_backbone_dependent
    assert "ALA" in lib.residues
    # Every non-Gly AA has a wildcard fallback bin.
    confs, weights = lib.rotamers_for("ARG", phi=None, psi=None)
    assert confs.shape[0] == weights.size > 0
