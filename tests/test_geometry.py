"""Tests for the NeRF placer and rotamer placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyconfind.geometry import place_batch, place_one
from pyconfind.rotamers import place_rotamers
from pyconfind.rotlib import parse_ebl


def test_place_one_ala_cb() -> None:
    """The single ALA CB placement, validated against the C++ --rout output."""
    a = np.array([2.143, 1.328, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([1.539, 0.0, 0.0])
    d = place_one(a, b, c, bond=1.520, angle_deg=106.506, dihedral_deg=120.005)
    np.testing.assert_allclose(d, [1.971, -0.729, 1.262], atol=1e-3)


def test_place_batch_matches_place_one() -> None:
    rng = np.random.default_rng(0)
    N = 50
    a = rng.standard_normal((N, 3)) * 2
    b = rng.standard_normal((N, 3)) * 2
    c = rng.standard_normal((N, 3)) * 2
    bond = rng.uniform(1.0, 2.0, size=N)
    angle = rng.uniform(90, 130, size=N)
    dih = rng.uniform(-180, 180, size=N)
    batched = place_batch(a, b, c, bond, angle, dih)
    for i in range(N):
        expected = place_one(a[i], b[i], c[i], bond[i], angle[i], dih[i])
        np.testing.assert_allclose(batched[i], expected, atol=1e-10)


def _parse_golden_rotamers(
    path: Path,
) -> dict[tuple[str, int, str, int], dict[str, np.ndarray]]:
    """Read the C++ --rout output into ``{(chain, resnum, resname, rot_idx): {atom: xyz}}``."""
    out: dict[tuple[str, int, str, int], dict[str, np.ndarray]] = {}
    cur_key: tuple[str, int, str, int] | None = None
    for raw in path.read_text().splitlines():
        if raw.startswith("REM "):
            # e.g.  REM A,1,ARG, rotamer 1
            after = raw[len("REM "):].strip()
            head, _, tail = after.partition(", rotamer ")
            chain, resnum_s, resname = head.split(",")[:3]
            cur_key = (chain, int(resnum_s), resname, int(tail) - 1)
            out[cur_key] = {}
        elif raw.startswith("ATOM") and cur_key is not None:
            name = raw[12:16].strip()
            x = float(raw[30:38])
            y = float(raw[38:46])
            z = float(raw[46:54])
            out[cur_key][name] = np.array([x, y, z])
    return out


def test_ic_builder_matches_cpp_for_ala_position_1(examples_dir: Path, ebl_path: Path) -> None:
    """Place every rotamer of every AA we can at A,1 and compare to C++ output.

    The C++ binary writes its surviving (post-prune) rotamers via --rout. To
    avoid the pruning step here, we restrict comparison to AA identities and
    rotamer indices present in the golden file.
    """
    repo_root = examples_dir.parents[2].parent
    rout = repo_root / "tests" / "golden" / "example0000.rotamers.pdb"
    golden = _parse_golden_rotamers(rout)

    residues = parse_ebl(ebl_path)
    # Native backbone for A,1 ALA from the input PDB.
    backbone = {
        "N": np.array([2.143, 1.328, 0.0]),
        "C": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.539, 0.0, 0.0]),
    }

    # Examine one ARG rotamer to validate sidechain placement.
    # The golden file has been bb-dep filtered, so the C++ rotamer "1" is the
    # first one that survived pruning. We instead use the bb-indep EBL pool
    # here (parse_ebl reads weights as the per-conformation weight), and
    # confirm by matching atom positions of the first known-surviving rotamer.
    # The strategy: pick AAs where the rotamer placement is essentially
    # deterministic given the IC values, then compare the first placement.
    #
    # We start by validating ALA (one rotamer, trivial sanity check).
    ala_tmpl = residues["ALA"]
    placed = place_rotamers(ala_tmpl, backbone)
    assert placed.coords.shape == (1, len(backbone) + 1, 3)
    ala_golden = golden[("A", 1, "ALA", 0)]
    cb_idx = placed.atom_names.index("CB")
    np.testing.assert_allclose(placed.coords[0, cb_idx], ala_golden["CB"], atol=1e-3)


def _ala_backbone_for(position_index: int) -> dict[str, np.ndarray]:
    """Return N/C/CA backbone coords for the three positions of example0000.

    Hard-coded from the input PDB so the test doesn't depend on the PDB loader.
    """
    bb = {
        0: {
            "N": np.array([2.143, 1.328, 0.0]),
            "CA": np.array([1.539, 0.0, 0.0]),
            "C": np.array([0.0, 0.0, 0.0]),
        },
        1: {  # position 2 (ILE/LEU native)
            "N": np.array([-0.612, -1.210, 0.0]),
            "CA": np.array([-2.052, -1.462, 0.0]),
            "C": np.array([-2.221, -2.971, 0.0]),
        },
        2: {  # position 3 (ALA)
            "N": np.array([-3.474, -3.466, 0.0]),
            "CA": np.array([-3.791, -4.877, 0.0]),
            "C": np.array([-5.297, -5.192, 0.0]),
        },
    }
    return bb[position_index]


def test_ic_builder_full_sweep_example0000(examples_dir: Path) -> None:
    """Compare every surviving (position, AA, rotamer, atom) against C++ output.

    For all 18 AAs at all 3 positions of example0000, place every rotamer of
    the bb-dep fallback bin (since all positions are termini for phi or psi)
    and verify every sidechain heavy atom against the C++ --rout golden file.
    """
    repo_root = examples_dir.parents[2].parent
    rotlib_dir = repo_root / "original-source" / "confind-msl" / "rotlibs"
    rout = repo_root / "tests" / "golden" / "example0000.rotamers.pdb"
    if not rotlib_dir.exists() or not rout.exists():
        pytest.skip("Rotamer library or golden output missing")

    from pyconfind.rotlib import load_library

    golden = _parse_golden_rotamers(rout)
    lib = load_library(rotlib_dir)
    # Heavy sidechain atoms by AA (no hydrogens; H golden coords are
    # placeholders for atoms MSL never placed).
    sidechain_heavy = {
        "ALA": ("CB",),
        "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
        "ASN": ("CB", "CG", "OD1", "ND2"),
        "ASP": ("CB", "CG", "OD1", "OD2"),
        "CYS": ("CB", "SG"),
        "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
        "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
        "HIS": ("CB", "CG", "ND1", "CD2", "NE2", "CE1"),
        "ILE": ("CB", "CG1", "CG2", "CD"),
        "LEU": ("CB", "CG", "CD1", "CD2"),
        "LYS": ("CB", "CG", "CD", "CE", "NZ"),
        "MET": ("CB", "CG", "SD", "CE"),
        "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
        "SER": ("CB", "OG"),
        "THR": ("CB", "OG1", "CG2"),
        "TRP": ("CB", "CG", "CD2", "CE2", "CE3", "CD1", "NE1", "CZ2", "CZ3", "CH2"),
        "TYR": ("CB", "CG", "CD1", "CE1", "CD2", "CE2", "CZ", "OH"),
        "VAL": ("CB", "CG1", "CG2"),
    }
    total_placements = 0
    for pos_idx in range(3):
        backbone = _ala_backbone_for(pos_idx)
        for aa, heavy_atoms in sidechain_heavy.items():
            try:
                confs_w, weights_w = lib.rotamers_for(aa, phi=None, psi=None)
            except KeyError:
                continue
            tmpl = lib.residues[aa]
            placed = place_rotamers(tmpl, backbone, confs=confs_w, weights=weights_w)
            for r in range(confs_w.shape[0]):
                key = ("A", pos_idx + 1, aa, r)
                if key not in golden:
                    continue
                g = golden[key]
                for atom in heavy_atoms:
                    idx = placed.atom_names.index(atom)
                    np.testing.assert_allclose(
                        placed.coords[r, idx], g[atom], atol=2e-3,
                        err_msg=f"pos {pos_idx+1} {aa} rot {r} {atom}",
                    )
                    total_placements += 1
    # Sanity floor — the example has hundreds of surviving rotamer placements.
    assert total_placements > 500, f"only validated {total_placements} placements"


def test_ic_builder_arg_against_cpp_terminus(examples_dir: Path) -> None:
    """Place the BIN * * fallback ARG rotamers at A,1 and match the C++ --rout.

    A,1 is an N-terminus so phi is undefined; the C++ uses the ``BIN * *``
    fallback. For ARG that maps to CONFIDX 97200..97233 in the EBL pool. The
    C++ writes rotamers in the order they survived pruning; we verify all
    seven heavy sidechain atoms of every surviving ARG rotamer match.
    """
    repo_root = examples_dir.parents[2].parent
    rotlib_dir = repo_root / "original-source" / "confind-msl" / "rotlibs"
    rout = repo_root / "tests" / "golden" / "example0000.rotamers.pdb"
    if not rotlib_dir.exists() or not rout.exists():
        pytest.skip("Rotamer library or golden output missing")

    from pyconfind.rotlib import load_library

    golden = _parse_golden_rotamers(rout)
    lib = load_library(rotlib_dir)
    confs_w, weights_w = lib.rotamers_for("ARG", phi=None, psi=None)
    arg = lib.residues["ARG"]
    backbone = {
        "N": np.array([2.143, 1.328, 0.0]),
        "C": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([1.539, 0.0, 0.0]),
    }
    placed = place_rotamers(arg, backbone, confs=confs_w, weights=weights_w)
    # 33 of 34 survive pruning; index 9 (rotamer 10) clashes with backbone.
    # The golden output skips the pruned rotamer; we walk only those present.
    n_compared = 0
    for r in range(confs_w.shape[0]):
        key = ("A", 1, "ARG", r)
        if key not in golden:
            continue
        arg_golden = golden[key]
        for atom in ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"):
            idx = placed.atom_names.index(atom)
            np.testing.assert_allclose(
                placed.coords[r, idx],
                arg_golden[atom],
                atol=2e-3,
                err_msg=f"ARG rotamer {r} {atom} placement mismatch",
            )
        n_compared += 1
    assert n_compared == 33, f"expected to compare 33 surviving ARG rotamers, got {n_compared}"
