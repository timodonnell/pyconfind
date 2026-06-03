"""Smoke tests for Analysis.positions_dataframe / contacts_dataframe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyconfind import analyze
from pyconfind.rotlib import load_library

ROTLIB = Path(__file__).resolve().parent / "data" / "mini_rotlib"
PDB = Path(__file__).resolve().parent / "data" / "structures" / "1UBQ.pdb"


def test_positions_dataframe_shape_and_columns() -> None:
    a = analyze(PDB, rotamer_library=load_library(ROTLIB))
    df = a.positions_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(a.positions)
    # Expected columns are present + per-position scores agree with the report.
    expected = {
        "chain", "resnum", "icode", "resname",
        "sumcond", "crwdnes", "freedom",
        "n_rotamers", "n_rotamers_placed", "fraction_pruned", "in_focus",
    }
    assert expected.issubset(df.columns)
    assert (df["sumcond"].to_numpy() == a.report.sum_contact_degree).all()
    assert (df["crwdnes"].to_numpy() == a.report.crwdnes).all()


def test_contacts_dataframe_matches_report() -> None:
    a = analyze(PDB, rotamer_library=load_library(ROTLIB))
    cdf = a.contacts_dataframe()
    assert isinstance(cdf, pd.DataFrame)
    assert len(cdf) == len(a.report.contacts)
    # Each row's chain/resnum/resname matches the position the contact points at.
    for row, c in zip(cdf.itertuples(index=False), a.report.contacts, strict=True):
        pi = a.positions[c.pos_i].position
        pj = a.positions[c.pos_j].position
        assert (row.chain_i, row.resnum_i, row.resname_i) == (pi.chain, pi.resnum, pi.resname)
        assert (row.chain_j, row.resnum_j, row.resname_j) == (pj.chain, pj.resnum, pj.resname)
        assert row.degree == c.degree
