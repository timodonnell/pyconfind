"""Byte-identity of pyconfind output vs the C++ reference, on real PDB/mmCIF.

Validation gate: the full text output must match the committed C++ ``.cont``
golden for each real structure (skips when the production library is absent,
e.g. in CI — the bundled-fixture pipeline tests cover that case)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyconfind import analyze, format_confind_text
from pyconfind.output import OutputOptions
from tests.conftest import REAL_STRUCTURES


@pytest.mark.parametrize("name", REAL_STRUCTURES)
@pytest.mark.parametrize("fmt", ["pdb", "cif"])
def test_output_byte_identical_to_cpp(
    structures_dir: Path, golden_dir: Path, rotlib_dir: Path, name: str, fmt: str
) -> None:
    golden = golden_dir / f"{name}.cont"
    if not golden.exists():
        pytest.skip(f"golden missing: {golden}")
    a = analyze(structures_dir / f"{name}.{fmt}", rotamer_library=rotlib_dir)
    assert format_confind_text(a.positions, a.report) == golden.read_text(), (
        f"{name}.{fmt} differs from the C++ reference"
    )


def test_pp_omg_columns(structures_dir: Path, rotlib_dir: Path) -> None:
    """The --pp/--omg style columns are emitted with phi/psi/omega values."""
    a = analyze(structures_dir / "1CRN.pdb", rotamer_library=rotlib_dir)
    text = format_confind_text(
        a.positions, a.report, OutputOptions(include_phi_psi=True, include_omega=True)
    )
    # An internal residue has all of phi, psi, omega defined (6 tab fields:
    # tag, pos, value, phi, psi, omega, resname).
    sumcond = [ln for ln in text.splitlines() if ln.startswith("sumcond")]
    internal = sumcond[len(sumcond) // 2].split("\t")
    assert len(internal) == 7
    for v in internal[3:6]:
        float(v)  # parseable phi/psi/omega
