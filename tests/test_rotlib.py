"""Tests for the rotamer library parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind.rotlib import (
    _bin_key,
    load_library,
    parse_bebl,
    parse_ebl,
)


def test_bin_key_matches_msl_rounding() -> None:
    # MSL: bin label = width * trunc((angle ± width/2) / width)
    assert _bin_key(-180.0, 10.0) == -180
    assert _bin_key(-175.0, 10.0) == -180
    assert _bin_key(-170.0, 10.0) == -170
    assert _bin_key(0.0, 10.0) == 0
    assert _bin_key(170.0, 10.0) == 170
    assert _bin_key(174.99, 10.0) == 170
    # phi == 180 rounds to bin 180, which is NOT in BEBL → caller falls back.
    assert _bin_key(180.0, 10.0) == 180


def test_parse_ebl_ala(ebl_path: Path) -> None:
    residues = parse_ebl(ebl_path)
    assert "ALA" in residues
    ala = residues["ALA"]
    assert ala.placed == ("CB",)
    # Only one CB, placed by dihedral N-C-CA-CB, angle C-CA-CB, length CA-CB
    np.testing.assert_array_equal(ala.parents, np.array([["N", "C", "CA"]]))
    assert ala.confs.shape == (1, 1, 3)
    np.testing.assert_allclose(ala.confs[0, 0], [120.005, 106.506, 1.520])
    np.testing.assert_allclose(ala.weights, [1.000])


def test_parse_ebl_arg_shape(ebl_path: Path) -> None:
    residues = parse_ebl(ebl_path)
    arg = residues["ARG"]
    assert arg.placed == (
        "CB", "CG", "CD", "NE", "CZ", "HE", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22",
    )
    M = len(arg.placed)
    assert arg.parents.shape == (M, 3)
    assert arg.confs.shape[1:] == (M, 3)
    # Weights and confs counts match
    assert arg.confs.shape[0] == arg.weights.size
    assert arg.confs.shape[0] == 97234


def test_parse_ebl_pro_placement_order(ebl_path: Path) -> None:
    """PRO has MOBI != placement order — CD is placed last in the open-chain build."""
    residues = parse_ebl(ebl_path)
    pro = residues["PRO"]
    assert pro.placed == ("CB", "CG", "CD")
    # CB: dihedral N-C-CA-CB, angle C-CA-CB, length CA-CB
    np.testing.assert_array_equal(pro.parents[0], ["N", "C", "CA"])
    # CD: dihedral CA-CB-CG-CD, angle CB-CG-CD, length CG-CD
    np.testing.assert_array_equal(pro.parents[2], ["CA", "CB", "CG"])


def test_parse_ebl_arg_first_conf_values(ebl_path: Path) -> None:
    residues = parse_ebl(ebl_path)
    arg = residues["ARG"]
    # First CONF row: 12 dihedrals, 12 angles, 12 bond lengths
    # From EBL.out line 50:
    #   CONF 120.017 62.5 176.9 176.6 85.7 -179.975 -179.987 -179.942 179.900 180.000
    #        179.942 -179.940 109.492 112.469 109.967 111.016 119.994 120.051 120.000
    #        119.972 120.009 120.010 120.038 119.948 1.520 1.520 1.520 1.450 1.330
    #        0.980 1.330 1.000 1.000 1.330 1.000 1.000
    first = arg.confs[0]
    # CB: dihedral 120.017, angle 109.492, length 1.520
    np.testing.assert_allclose(first[0], [120.017, 109.492, 1.520])
    # CG: dihedral 62.5, angle 112.469, length 1.520
    np.testing.assert_allclose(first[1], [62.5, 112.469, 1.520])
    # HE: dihedral -179.975, angle 120.051, length 0.980  (index 5)
    np.testing.assert_allclose(first[5], [-179.975, 120.051, 0.980])
    # HH22 last: dihedral -179.940, angle 119.948, length 1.000
    np.testing.assert_allclose(first[-1], [-179.940, 119.948, 1.000])


def test_parse_ebl_all_amino_acids(ebl_path: Path) -> None:
    residues = parse_ebl(ebl_path)
    expected = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
    assert expected.issubset(residues.keys())


def test_parse_bebl_header(bebl_path: Path) -> None:
    phi_bin, psi_bin, idx = parse_bebl(bebl_path)
    assert phi_bin == 10.0
    assert psi_bin == 10.0
    # ALA bin (-180,-180) -> all CONFIDX 0 (only one ALA rotamer)
    np.testing.assert_array_equal(idx[("ALA", -180, -180)], [0])
    # ARG bin (-180,-180) -> 75 CONFIDX values
    arg_bin = idx[("ARG", -180, -180)]
    assert arg_bin.size == 75
    np.testing.assert_array_equal(arg_bin[:5], [0, 1, 2, 3, 4])


def test_load_library_bb_dep(rotlib_dir: Path) -> None:
    lib = load_library(rotlib_dir)
    assert lib.is_backbone_dependent
    assert lib.phi_bin == 10.0
    assert lib.psi_bin == 10.0
    confs, weights = lib.rotamers_for("ARG", phi=-180.0, psi=-180.0)
    assert confs.shape[0] == 75
    assert weights.size == 75


def test_load_library_phi_180_falls_back_to_wildcard(rotlib_dir: Path) -> None:
    """phi=180 rounds to bin 180, which is not in BEBL. MSL falls back to
    ``BIN * *`` — we must too."""
    lib = load_library(rotlib_dir)
    confs_180, _ = lib.rotamers_for("ARG", phi=180.0, psi=180.0)
    confs_wild, _ = lib.rotamers_for("ARG", phi=None, psi=None)
    assert confs_180.shape == confs_wild.shape
    np.testing.assert_array_equal(confs_180, confs_wild)


def test_load_library_rejects_single_file(ebl_path: Path) -> None:
    """Backbone-independent (single-file) libraries are intentionally
    unsupported — they would silently misuse the bb-dep conditional weights."""
    with pytest.raises(ValueError, match="not a directory"):
        load_library(ebl_path)


def test_gly_has_no_rotamers(rotlib_dir: Path) -> None:
    """GLY is in the EBL pool but absent from BEBL; rotamers_for returns empty
    rather than raising (regression for the --native-only GLY crash)."""
    lib = load_library(rotlib_dir)
    confs, weights = lib.rotamers_for("GLY", phi=None, psi=None)
    assert confs.shape[0] == 0
    assert weights.size == 0
    # A real bb-dep AA still returns rotamers.
    confs2, _ = lib.rotamers_for("LEU", phi=-60.0, psi=-45.0)
    assert confs2.shape[0] > 0


def test_load_library_missing_files(tmp_path: Path) -> None:
    # Directory with no EBL.out should raise
    with pytest.raises(FileNotFoundError):
        load_library(tmp_path)
