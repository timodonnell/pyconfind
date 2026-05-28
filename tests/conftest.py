"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
# The full production rotamer library (100+ MB) is not committed; it lives
# under the local original-source/ tree extracted from the upstream tarball.
ROTLIB_DIR = _REPO / "original-source" / "confind-msl" / "rotlibs"
# Example PDBs are bundled under tests/data/examples so they are available in
# CI; fall back to the original-source copy if the bundled set is absent.
_BUNDLED_EXAMPLES = Path(__file__).resolve().parent / "data" / "examples"
_SRC_EXAMPLES = (
    _REPO / "original-source" / "confind-msl" / "mslib" / "exampleFiles"
)
EXAMPLES_DIR = _BUNDLED_EXAMPLES if _BUNDLED_EXAMPLES.exists() else _SRC_EXAMPLES


@pytest.fixture(scope="session")
def rotlib_dir() -> Path:
    if not ROTLIB_DIR.exists():
        pytest.skip(f"Full rotamer library not found at {ROTLIB_DIR}")
    return ROTLIB_DIR


@pytest.fixture(scope="session")
def ebl_path(rotlib_dir: Path) -> Path:
    return rotlib_dir / "EBL.out"


@pytest.fixture(scope="session")
def bebl_path(rotlib_dir: Path) -> Path:
    return rotlib_dir / "BEBL.out"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    if not EXAMPLES_DIR.exists():
        pytest.skip(f"Example PDBs not found at {EXAMPLES_DIR}")
    return EXAMPLES_DIR
