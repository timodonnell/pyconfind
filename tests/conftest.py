"""Shared pytest fixtures.

Tests run against **real PDB structures** committed under
``tests/data/structures/`` (downloaded from the RCSB), with C++-reference
contact maps in ``tests/golden/``. The structures are small but cover the
cases that matter: GLY/PRO (1CRN, 1UBQ), alternate locations (1EJG), and both
PDB and mmCIF inputs.

The full 100+ MB production rotamer library is not committed; byte-identity
tests against the C++ golden self-skip when it is absent (it lives under
``original-source/`` locally). CI-runnable tests use the bundled mini library.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent

#: Real structures committed for testing (RCSB ids), with notable features.
REAL_STRUCTURES = ("1CRN", "1UBQ", "1EJG")

STRUCTURES_DIR = _HERE / "data" / "structures"
GOLDEN_DIR = _HERE / "golden"
MINI_ROTLIB = _HERE / "data" / "mini_rotlib"

# Full production library (backbone-dependent) — local only, not committed.
ROTLIB_DIR = _REPO / "original-source" / "confind-msl" / "rotlibs"


@pytest.fixture(scope="session")
def structures_dir() -> Path:
    if not STRUCTURES_DIR.exists():
        pytest.skip(f"real structures not found at {STRUCTURES_DIR}")
    return STRUCTURES_DIR


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def mini_rotlib() -> Path:
    if not MINI_ROTLIB.exists():
        pytest.skip(f"mini rotamer library not found at {MINI_ROTLIB}")
    return MINI_ROTLIB


@pytest.fixture(scope="session")
def rotlib_dir() -> Path:
    """The full production library; skips if not present (e.g. in CI)."""
    if not ROTLIB_DIR.exists():
        pytest.skip(f"full rotamer library not found at {ROTLIB_DIR}")
    return ROTLIB_DIR


@pytest.fixture(scope="session")
def ebl_path(rotlib_dir: Path) -> Path:
    return rotlib_dir / "EBL.out"


@pytest.fixture(scope="session")
def bebl_path(rotlib_dir: Path) -> Path:
    return rotlib_dir / "BEBL.out"
