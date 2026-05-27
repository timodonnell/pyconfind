"""Tests for the library API + CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyconfind import analyze, format_confind_text, format_json


def test_analyze_returns_full_pipeline(examples_dir: Path, rotlib_dir: Path) -> None:
    res = analyze(examples_dir / "example0000.pdb", rotamer_library=rotlib_dir)
    assert len(res.positions) == 3
    assert res.positions[0].position.resname == "ALA"
    # Smoke: contact-degree numbers sane.
    assert res.report.contacts
    # Smoke: text output renders.
    text = format_confind_text(res.positions, res.report)
    assert "contact\tA,1\tA,3\t0.009889" in text


def test_analyze_native_only_skips_substitutions(
    examples_dir: Path, rotlib_dir: Path
) -> None:
    """With --native-only, each position should only have rotamers of its
    native AA — and far fewer rotamers total."""
    full = analyze(examples_dir / "example0000.pdb", rotamer_library=rotlib_dir)
    native = analyze(
        examples_dir / "example0000.pdb",
        rotamer_library=rotlib_dir,
        native_only=True,
    )
    for pr_full, pr_native in zip(full.positions, native.positions, strict=True):
        # All native rotamers should be of the native residue.
        for rot in pr_native.rotamers:
            assert rot.aa == pr_native.position.resname
        # Should never have more rotamers than the full run did.
        assert len(pr_native.rotamers) <= len(pr_full.rotamers)


def test_json_output_smokes(examples_dir: Path, rotlib_dir: Path) -> None:
    res = analyze(examples_dir / "example0000.pdb", rotamer_library=rotlib_dir)
    payload = json.loads(format_json(res.positions, res.report))
    assert payload["sequence"] == ["ALA", "ILE", "ALA"]
    assert len(payload["positions"]) == 3
    assert len(payload["contacts"]) >= 2


def test_cli_text_output_matches_cpp(
    examples_dir: Path, rotlib_dir: Path, tmp_path: Path
) -> None:
    """Invoke the installed CLI end-to-end and diff its output against the
    C++ binary's golden output."""
    pdb = examples_dir / "example0002.pdb"
    out = tmp_path / "out.cont"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(pdb),
            "--rLib", str(rotlib_dir),
            "--o", str(out),
        ],
        check=True,
        capture_output=True,
    )
    golden = Path(__file__).resolve().parent / "golden" / "example0002.cont"
    assert out.read_text() == golden.read_text()


def test_cli_json_output(examples_dir: Path, rotlib_dir: Path, tmp_path: Path) -> None:
    pdb = examples_dir / "example0000.pdb"
    out = tmp_path / "out.json"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(pdb),
            "--rLib", str(rotlib_dir),
            "--o", str(out),
            "--json",
        ],
        check=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text())
    assert "positions" in payload
    assert "contacts" in payload


def test_cli_requires_input(rotlib_dir: Path) -> None:
    """Without --p or --pL the CLI should error out with a usage message
    mentioning the missing flag."""
    result = subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--rLib", str(rotlib_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = (result.stderr or "") + (result.stdout or "")
    assert "--p" in combined or "--pL" in combined


def test_cli_pp_flag(examples_dir: Path, rotlib_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.cont"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(examples_dir / "example0000.pdb"),
            "--rLib", str(rotlib_dir),
            "--o", str(out),
            "--pp",
        ],
        check=True,
        capture_output=True,
    )
    text = out.read_text()
    # phi/psi columns should appear in sumcond rows
    assert "sumcond\tA,2\t0.000000\t180.000000\t180.000000\tILE" in text


@pytest.mark.parametrize(
    "pdb_name",
    [
        "example0000.pdb",
        "example0002.pdb",
        "example0008.pdb",
    ],
)
def test_cli_end_to_end_byte_identical(
    examples_dir: Path, rotlib_dir: Path, tmp_path: Path, pdb_name: str
) -> None:
    """The CLI's output must match the C++ binary's output byte-for-byte."""
    pdb = examples_dir / pdb_name
    golden = Path(__file__).resolve().parent / "golden" / (pdb.stem + ".cont")
    out = tmp_path / "out.cont"
    subprocess.run(
        [
            sys.executable, "-m", "pyconfind.cli",
            "--p", str(pdb),
            "--rLib", str(rotlib_dir),
            "--o", str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert out.read_text() == golden.read_text(), (
        f"CLI output for {pdb_name} differs from C++ reference"
    )
