"""Byte-identity tests against the reference C++ output."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyconfind.build import build_position_rotamers
from pyconfind.contacts import compute_contacts
from pyconfind.output import OutputOptions, format_confind_text, format_json
from pyconfind.pdb import read_pdb
from pyconfind.rotlib import load_library
from pyconfind.structure import positions_from_atoms


@pytest.mark.parametrize(
    "pdb_name",
    [
        "example0000.pdb",
        "example0001.pdb",
        "example0002.pdb",
        "example0003.pdb",
        "example0004.pdb",
        "example0005.pdb",
        "example0006.pdb",
        "example0007.pdb",
        "example0008.pdb",
        "example0008_caOnly.pdb",
        "example0009_caOnly.pdb",
    ],
)
def test_text_output_byte_identical_to_cpp(
    examples_dir: Path, rotlib_dir: Path, pdb_name: str
) -> None:
    """The full text output must match the C++ binary's output byte-for-byte."""
    pdb_path = examples_dir / pdb_name
    golden_path = (
        Path(__file__).resolve().parent / "golden" / (pdb_path.stem + ".cont")
    )
    if not golden_path.exists():
        pytest.skip(f"golden output missing: {golden_path}")
    atoms = read_pdb(pdb_path)
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    report = compute_contacts(result)
    mine = format_confind_text(result, report)
    assert mine == golden_path.read_text(), (
        f"output for {pdb_name} differs from C++ reference"
    )


def test_json_output_round_trips(examples_dir: Path, rotlib_dir: Path) -> None:
    """Sanity check on the JSON formatter."""
    import json

    atoms = read_pdb(examples_dir / "example0000.pdb")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    report = compute_contacts(result)
    j = format_json(result, report)
    payload = json.loads(j)
    assert "positions" in payload
    assert "contacts" in payload
    assert payload["sequence"] == ["ALA", "ILE", "ALA"]
    # The contact A,1-A,3 with degree 0.009889 should be in the output.
    found = next(
        (
            c for c in payload["contacts"]
            if c["i"]["resnum"] == 1 and c["j"]["resnum"] == 3
        ),
        None,
    )
    assert found is not None
    assert abs(found["degree"] - 0.009889) < 1e-5


def test_pp_omg_flags_match_cpp(examples_dir: Path, rotlib_dir: Path) -> None:
    """Verify the --pp and --omg style output options against C++ output for
    example0000 (we have an --pp run logged earlier)."""
    atoms = read_pdb(examples_dir / "example0000.pdb")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    report = compute_contacts(result)
    mine = format_confind_text(
        result, report, OutputOptions(include_phi_psi=True)
    )
    # From the C++ --pp run earlier in this session:
    #   sumcond A,1 0.009889 999.000000 180.000000 ALA
    #   sumcond A,2 0.000000 180.000000 180.000000 ILE
    #   sumcond A,3 0.009889 180.000000 999.000000 ALA
    assert "sumcond\tA,1\t0.009889\t999.000000\t180.000000\tALA" in mine
    assert "sumcond\tA,2\t0.000000\t180.000000\t180.000000\tILE" in mine
    assert "sumcond\tA,3\t0.009889\t180.000000\t999.000000\tALA" in mine
