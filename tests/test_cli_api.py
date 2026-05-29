"""Library API + CLI, on real structures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyconfind import analyze, format_confind_text, format_json
from tests.conftest import REAL_STRUCTURES


def test_analyze_returns_full_pipeline(structures_dir: Path, mini_rotlib: Path) -> None:
    res = analyze(structures_dir / "1CRN.pdb", rotamer_library=mini_rotlib)
    assert len(res.positions) == 46  # crambin
    assert res.report.contacts
    lines = format_confind_text(res.positions, res.report).splitlines()
    assert lines[0].startswith("contact\t")
    assert lines[-1].startswith("SEQUENCE:")


def test_json_output_smokes(structures_dir: Path, mini_rotlib: Path) -> None:
    res = analyze(structures_dir / "1UBQ.pdb", rotamer_library=mini_rotlib)
    payload = json.loads(format_json(res.positions, res.report))
    assert len(payload["sequence"]) == len(res.positions)
    assert len(payload["positions"]) == len(res.positions)
    assert payload["contacts"]


def test_cli_json_output(structures_dir: Path, mini_rotlib: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, "-m", "pyconfind.cli", "--p", str(structures_dir / "1CRN.pdb"),
         "--rLib", str(mini_rotlib), "--o", str(out), "--json"],
        check=True, capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert "positions" in payload and "contacts" in payload


def test_cli_requires_input(mini_rotlib: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pyconfind.cli", "--rLib", str(mini_rotlib)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr or "") + (result.stdout or "")
    assert "--p" in combined or "--pL" in combined


def test_cli_pp_flag(structures_dir: Path, mini_rotlib: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.cont"
    subprocess.run(
        [sys.executable, "-m", "pyconfind.cli", "--p", str(structures_dir / "1CRN.pdb"),
         "--rLib", str(mini_rotlib), "--o", str(out), "--pp"],
        check=True, capture_output=True,
    )
    # sumcond rows gain phi/psi columns: tag, pos, value, phi, psi, resname.
    sumcond = [ln for ln in out.read_text().splitlines() if ln.startswith("sumcond")]
    assert sumcond and len(sumcond[len(sumcond) // 2].split("\t")) == 6


def test_cli_mmcif_input(structures_dir: Path, mini_rotlib: Path, tmp_path: Path) -> None:
    """The CLI accepts mmCIF input and gives the same output as PDB."""
    out_cif = tmp_path / "cif.cont"
    out_pdb = tmp_path / "pdb.cont"
    for src, dst in [("1CRN.cif", out_cif), ("1CRN.pdb", out_pdb)]:
        subprocess.run(
            [sys.executable, "-m", "pyconfind.cli", "--p", str(structures_dir / src),
             "--rLib", str(mini_rotlib), "--o", str(dst)],
            check=True, capture_output=True,
        )
    assert out_cif.read_text() == out_pdb.read_text()


@pytest.mark.parametrize("name", REAL_STRUCTURES)
def test_cli_end_to_end_byte_identical(
    structures_dir: Path, golden_dir: Path, rotlib_dir: Path, tmp_path: Path, name: str
) -> None:
    """CLI output must match the C++ golden byte-for-byte (full library)."""
    golden = golden_dir / f"{name}.cont"
    if not golden.exists():
        pytest.skip("golden missing")
    out = tmp_path / "out.cont"
    subprocess.run(
        [sys.executable, "-m", "pyconfind.cli", "--p", str(structures_dir / f"{name}.pdb"),
         "--rLib", str(rotlib_dir), "--o", str(out)],
        check=True, capture_output=True,
    )
    assert out.read_text() == golden.read_text(), f"CLI output for {name} differs"
