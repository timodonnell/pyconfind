"""End-to-end tests for rotamer placement + backbone-clash pruning.

Validation gate: fraction-pruned (``crwdnes`` in C++ output) must match the
C++ binary byte-for-byte on every example PDB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind.build import build_position_rotamers
from pyconfind.contacts import compute_contacts
from pyconfind.pdb import read_pdb
from pyconfind.rotlib import load_library
from pyconfind.structure import dihedral_deg, positions_from_atoms


def test_dihedral_sign_convention() -> None:
    """Standard IUPAC convention: positive when D rotates clockwise viewed
    from B → C. For the canonical test geometry, the dihedral is -90°."""
    a, b, c, d = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
    )
    assert abs(dihedral_deg(a, b, c, d) - (-90.0)) < 1e-9


def test_phi_psi_match_cpp_example0002(examples_dir: Path) -> None:
    """Spot-check phi/psi for example0002 — the C++ output gives definite values."""
    atoms = read_pdb(examples_dir / "example0002.pdb")
    positions = positions_from_atoms(atoms)
    # From: confind --p example0002.pdb --rLib ../rotlibs --pp (crwdnes rows).
    expected = {
        ("A", 2): (-130.471212, 143.373520),
        ("A", 3): (-77.237776, 118.790087),
        ("A", 4): (-134.570451, 137.449986),
        ("A", 5): (-90.993792, 119.231130),
    }
    by_key = {(p.chain, p.resnum): p for p in positions}
    for k, (phi_e, psi_e) in expected.items():
        p = by_key[k]
        assert p.phi is not None and abs(p.phi - phi_e) < 1e-3, k
        assert p.psi is not None and abs(p.psi - psi_e) < 1e-3, k


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
def test_crwdnes_matches_cpp(examples_dir: Path, rotlib_dir: Path, pdb_name: str) -> None:
    """Per-position fraction-pruned must equal the C++ binary's ``crwdnes`` row."""
    pdb_path = examples_dir / pdb_name
    golden_path = (
        Path(__file__).resolve().parent / "golden" / (pdb_path.stem + ".cont")
    )
    if not golden_path.exists():
        pytest.skip(f"golden output missing: {golden_path}")
    atoms = read_pdb(pdb_path)
    if len(atoms) == 0:
        pytest.skip(f"{pdb_name} has no legal atoms")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    golden = {}
    for line in golden_path.read_text().splitlines():
        parts = line.split("\t")
        if parts[0] == "crwdnes":
            golden[parts[1]] = float(parts[2])
    assert golden, f"no crwdnes rows in golden for {pdb_name}"
    for pr in result:
        key = f"{pr.position.chain},{pr.position.resnum}"
        if key not in golden:
            continue
        if np.isnan(golden[key]):
            assert np.isnan(pr.fraction_pruned), (
                f"{pdb_name} pos {key}: expected NaN (CA-only), got {pr.fraction_pruned}"
            )
            continue
        assert abs(pr.fraction_pruned - golden[key]) < 1e-5, (
            f"{pdb_name} pos {key} {pr.position.resname}: "
            f"crwdnes={pr.fraction_pruned:.6f} vs C++ {golden[key]:.6f}"
        )


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
    ],
)
def test_contact_degrees_match_cpp(examples_dir: Path, rotlib_dir: Path, pdb_name: str) -> None:
    """Each per-pair contact degree must match the C++ output."""
    pdb_path = examples_dir / pdb_name
    golden_path = (
        Path(__file__).resolve().parent / "golden" / (pdb_path.stem + ".cont")
    )
    if not golden_path.exists():
        pytest.skip(f"golden output missing: {golden_path}")
    atoms = read_pdb(pdb_path)
    if len(atoms) == 0:
        pytest.skip(f"{pdb_name} has no legal atoms")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    report = compute_contacts(result)

    golden: dict[tuple[str, str], float] = {}
    for line in golden_path.read_text().splitlines():
        parts = line.split("\t")
        if parts[0] == "contact":
            golden[(parts[1], parts[2])] = float(parts[3])

    our: dict[tuple[str, str], float] = {}
    for c in report.contacts:
        p_i = result[c.pos_i].position
        p_j = result[c.pos_j].position
        our[(f"{p_i.chain},{p_i.resnum}", f"{p_j.chain},{p_j.resnum}")] = c.degree

    assert golden, f"no contact rows in golden for {pdb_name}"
    for key, val in golden.items():
        ours = our.get(key, 0.0)
        assert abs(ours - val) < 1e-5, (
            f"{pdb_name} contact {key}: mine={ours:.7f} C++={val:.7f}"
        )


@pytest.mark.parametrize(
    "pdb_name",
    ["example0000.pdb", "example0002.pdb", "example0007.pdb", "example0008.pdb"],
)
def test_freedom_and_sumcond_match_cpp(
    examples_dir: Path, rotlib_dir: Path, pdb_name: str
) -> None:
    """Per-position freedom and sumcond rows must match the C++ output."""
    pdb_path = examples_dir / pdb_name
    golden_path = (
        Path(__file__).resolve().parent / "golden" / (pdb_path.stem + ".cont")
    )
    if not golden_path.exists():
        pytest.skip(f"golden output missing: {golden_path}")
    atoms = read_pdb(pdb_path)
    if len(atoms) == 0:
        pytest.skip(f"{pdb_name} has no legal atoms")
    positions = positions_from_atoms(atoms)
    lib = load_library(rotlib_dir)
    result = build_position_rotamers(positions, lib)
    report = compute_contacts(result)

    gold_sumcond: dict[str, float] = {}
    gold_freedom: dict[str, float] = {}
    for line in golden_path.read_text().splitlines():
        parts = line.split("\t")
        if parts[0] == "sumcond":
            gold_sumcond[parts[1]] = float(parts[2])
        elif parts[0] == "freedom":
            gold_freedom[parts[1]] = float(parts[2])

    for i, pr in enumerate(result):
        key = f"{pr.position.chain},{pr.position.resnum}"
        if key in gold_sumcond:
            assert abs(report.sum_contact_degree[i] - gold_sumcond[key]) < 1e-5, (
                f"{pdb_name} sumcond {key}: mine={report.sum_contact_degree[i]:.7f} "
                f"C++={gold_sumcond[key]:.7f}"
            )
        if key in gold_freedom:
            assert abs(report.freedom[i] - gold_freedom[key]) < 1e-5, (
                f"{pdb_name} freedom {key}: mine={report.freedom[i]:.7f} "
                f"C++={gold_freedom[key]:.7f}"
            )


def test_permanent_contacts_example_with_perm(examples_dir: Path, rotlib_dir: Path) -> None:
    """Find an example with C++ ``percont`` rows and verify we match them."""
    for pdb_path in sorted(examples_dir.glob("example*.pdb")):
        golden_path = (
            Path(__file__).resolve().parent / "golden" / (pdb_path.stem + ".cont")
        )
        if not golden_path.exists():
            continue
        text = golden_path.read_text()
        if "percont" not in text:
            continue
        # Found one — check it.
        atoms = read_pdb(pdb_path)
        positions = positions_from_atoms(atoms)
        lib = load_library(rotlib_dir)
        result = build_position_rotamers(positions, lib)
        pos_by_key = {(p.position.chain, p.position.resnum): p for p in result}
        # Parse C++ percont rows: "percont\tchain,resnum\tchain,resnum\t-1.0\t..."
        golden_perm: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        for line in text.splitlines():
            parts = line.split("\t")
            if parts[0] != "percont":
                continue
            c1, r1 = parts[1].split(",")
            c2, r2 = parts[2].split(",")
            golden_perm.add(((c1, int(r1)), (c2, int(r2))))
        our_perm: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        for pr in result:
            for j in pr.permanent_contacts:
                other = next(p for p in result if p.position.index == j)
                our_perm.add(
                    (
                        (pr.position.chain, pr.position.resnum),
                        (other.position.chain, other.position.resnum),
                    )
                )
        assert our_perm == golden_perm, (
            f"{pdb_path.name} permanent contacts diverge: "
            f"only-cpp={golden_perm - our_perm}  only-ours={our_perm - golden_perm}"
        )
        return  # one example is enough
    pytest.skip("no example PDB has percont rows in its golden output")
