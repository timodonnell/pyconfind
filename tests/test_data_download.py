"""Tests for the rotamer-library download helper.

Uses the bundled mini library packaged into a local tar.gz served via a
``file://`` URL — no network access required in CI.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from pyconfind import ROTAMER_LIBRARY_URL, download_rotamer_library

MINI_ROTLIB = Path(__file__).resolve().parent / "data" / "mini_rotlib"


@pytest.fixture()
def mini_tarball(tmp_path: Path) -> str:
    """Pack the mini library as ``rotlibs/{EBL,BEBL}.out`` and return a file URL."""
    tar_path = tmp_path / "rotlibs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(MINI_ROTLIB / "EBL.out", arcname="rotlibs/EBL.out")
        tar.add(MINI_ROTLIB / "BEBL.out", arcname="rotlibs/BEBL.out")
    return tar_path.as_uri()


def test_download_extracts_library(tmp_path: Path, mini_tarball: str) -> None:
    dest = tmp_path / "rotlibs"
    out = download_rotamer_library(dest=dest, url=mini_tarball)
    assert out == dest
    assert (dest / "EBL.out").exists()
    assert (dest / "BEBL.out").exists()


def test_download_is_idempotent(tmp_path: Path, mini_tarball: str) -> None:
    dest = tmp_path / "rotlibs"
    download_rotamer_library(dest=dest, url=mini_tarball)
    # Second call with a bogus URL must not re-download (dest already populated).
    out = download_rotamer_library(dest=dest, url="http://invalid.invalid/x.tar.gz")
    assert out == dest


def test_downloaded_library_loads(tmp_path: Path, mini_tarball: str) -> None:
    from pyconfind import load_library

    dest = download_rotamer_library(dest=tmp_path / "rotlibs", url=mini_tarball)
    lib = load_library(dest)
    assert lib.is_backbone_dependent
    assert "ALA" in lib.residues


def test_default_url_points_at_release() -> None:
    assert ROTAMER_LIBRARY_URL.startswith("https://github.com/")
    assert ROTAMER_LIBRARY_URL.endswith("rotlibs.tar.gz")
