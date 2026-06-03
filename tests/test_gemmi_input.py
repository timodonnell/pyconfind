"""analyze() can be handed a pre-parsed gemmi.Structure instead of a path.

The byte-equivalence check is the real assertion: path-input and
gemmi-input must produce indistinguishable output for the same structure.
"""

from __future__ import annotations

from pathlib import Path

import gemmi

from pyconfind import analyze, format_confind_text, load_library
from pyconfind.pdb import atoms_from_gemmi_structure, read_structure

MINI_ROTLIB = Path(__file__).resolve().parent / "data" / "mini_rotlib"
STRUCT = Path(__file__).resolve().parent / "data" / "structures"


def _toy_structure(chain_name: str = "A") -> gemmi.Structure:
    st = gemmi.Structure()
    st.name = "toy"
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId("1")
    for atom_name, element in [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]:
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(0.0, 0.0, 0.0)
        res.add_atom(atom)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    return st


def test_analyze_accepts_gemmi_structure() -> None:
    lib = load_library(MINI_ROTLIB)
    pdb = STRUCT / "1UBQ.pdb"

    via_path = analyze(pdb, rotamer_library=lib)
    via_gemmi = analyze(gemmi.read_structure(str(pdb)), rotamer_library=lib)

    assert format_confind_text(via_path.positions, via_path.report) == \
        format_confind_text(via_gemmi.positions, via_gemmi.report)


def test_atoms_from_gemmi_matches_read_structure() -> None:
    pdb = STRUCT / "5TRU.pdb"
    via_path = read_structure(pdb)
    via_gemmi = atoms_from_gemmi_structure(gemmi.read_structure(str(pdb)))

    assert len(via_path.chain) == len(via_gemmi.chain)
    assert (via_path.xyz == via_gemmi.xyz).all()
    assert (via_path.chain == via_gemmi.chain).all()


def test_gemmi_input_honors_assembly() -> None:
    """assembly= must still apply when input is a gemmi.Structure."""
    lib = load_library(MINI_ROTLIB)
    st = gemmi.read_structure(str(STRUCT / "5TRU.pdb"))
    one = analyze(st, rotamer_library=lib, native_only=True, assembly=1)
    two = analyze(st, rotamer_library=lib, native_only=True, assembly="2")
    chains_one = sorted({p.position.chain for p in one.positions})
    chains_two = sorted({p.position.chain for p in two.positions})
    assert chains_one == ["C", "H", "L"]
    assert chains_two == ["c", "h", "l"]


def test_gemmi_input_preserves_multi_character_chain_ids() -> None:
    atoms = atoms_from_gemmi_structure(_toy_structure("ABC"), assembly=None)
    assert atoms.chain.tolist() == ["ABC"] * 5
