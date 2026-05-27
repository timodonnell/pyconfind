"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

ROTLIB_DIR = Path(__file__).resolve().parents[1] / "original-source" / "confind-msl" / "rotlibs"
EXAMPLES_DIR = (
    Path(__file__).resolve().parents[1]
    / "original-source"
    / "confind-msl"
    / "mslib"
    / "exampleFiles"
)


@pytest.fixture(scope="session")
def rotlib_dir() -> Path:
    if not ROTLIB_DIR.exists():
        pytest.skip(f"Rotamer library not found at {ROTLIB_DIR}")
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
