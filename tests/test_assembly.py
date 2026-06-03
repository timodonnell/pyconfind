"""Bio-assembly handling.

5TRU's asymmetric unit contains two independent Fab-CTLA-4 complexes (chains
``L H C`` and ``l h c``). pyconfind defaults to ``assembly=1``, which restricts
analysis to a single complex; ``assembly=None`` keeps the full AU; ``=2`` picks
the other copy. Single-chain monomers (1CRN/1UBQ/1EJG) declare an identity
assembly, so the default is byte-safe for them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyconfind import read_structure

STRUCT = Path(__file__).resolve().parent / "data" / "structures"


def test_assembly_default_picks_first_for_5tru() -> None:
    atoms = read_structure(STRUCT / "5TRU.pdb")  # default assembly=1
    chains = sorted(set(atoms.chain.tolist()))
    assert chains == ["C", "H", "L"], chains


def test_assembly_two_picks_second_copy() -> None:
    atoms = read_structure(STRUCT / "5TRU.pdb", assembly="2")
    assert sorted(set(atoms.chain.tolist())) == ["c", "h", "l"]


def test_assembly_none_uses_full_au() -> None:
    atoms = read_structure(STRUCT / "5TRU.pdb", assembly=None)
    assert sorted(set(atoms.chain.tolist())) == ["C", "H", "L", "c", "h", "l"]


def test_assembly_unknown_raises() -> None:
    with pytest.raises(ValueError, match="assembly '99' not found"):
        read_structure(STRUCT / "5TRU.pdb", assembly=99)


def test_assembly_default_byte_safe_for_monomers() -> None:
    """Default ``assembly=1`` must not change the AU for monomeric goldens."""
    for pdb in ("1CRN", "1UBQ", "1EJG"):
        with_default = read_structure(STRUCT / f"{pdb}.pdb")
        as_au = read_structure(STRUCT / f"{pdb}.pdb", assembly=None)
        assert len(with_default.chain) == len(as_au.chain)
        assert (with_default.xyz == as_au.xyz).all()
