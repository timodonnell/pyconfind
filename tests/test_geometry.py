"""NeRF placement: self-consistency + validation against the C++ --rout output.

The direct placement check uses 1CRN's rotamer dump from the reference binary
(tests/golden/1CRN.rotamers.pdb). Skips when the full library is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconfind.geometry import place_batch, place_one
from pyconfind.pdb import read_structure
from pyconfind.rotamers import place_rotamers
from pyconfind.structure import positions_from_atoms

# Heavy sidechain atoms by AA (hydrogens excluded; H golden coords are
# placeholders for atoms MSL never placed).
SIDECHAIN_HEAVY = {
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
_DEFAULT_AAS = tuple(SIDECHAIN_HEAVY)


def test_place_batch_matches_place_one() -> None:
    """The vectorized placer must equal the scalar one for arbitrary geometry."""
    rng = np.random.default_rng(0)
    n = 50
    a = rng.standard_normal((n, 3)) * 2
    b = rng.standard_normal((n, 3)) * 2
    c = rng.standard_normal((n, 3)) * 2
    bond = rng.uniform(1.0, 2.0, size=n)
    angle = rng.uniform(90, 130, size=n)
    dih = rng.uniform(-180, 180, size=n)
    batched = place_batch(a, b, c, bond, angle, dih)
    for i in range(n):
        np.testing.assert_allclose(
            batched[i], place_one(a[i], b[i], c[i], bond[i], angle[i], dih[i]), atol=1e-10
        )


def _parse_rout(path: Path) -> dict[tuple[str, int, str, int], dict[str, np.ndarray]]:
    out: dict[tuple[str, int, str, int], dict[str, np.ndarray]] = {}
    cur: tuple[str, int, str, int] | None = None
    for raw in path.read_text().splitlines():
        if raw.startswith("REM "):
            head, _, tail = raw[4:].strip().partition(", rotamer ")
            chain, resnum_s, resname = head.split(",")[:3]
            cur = (chain, int(resnum_s), resname, int(tail) - 1)
            out[cur] = {}
        elif raw.startswith("ATOM") and cur is not None:
            out[cur][raw[12:16].strip()] = np.array(
                [float(raw[30:38]), float(raw[38:46]), float(raw[46:54])]
            )
    return out


def test_ic_builder_matches_cpp_rout(structures_dir: Path, rotlib_dir: Path) -> None:
    """Placed sidechain heavy atoms must match the C++ --rout dump for 1CRN.

    For every position and every substituted AA, place the rotamers from the
    backbone-dependent bin (the same the C++ used) and compare each surviving
    rotamer's heavy atoms to the reference coordinates.
    """
    from pyconfind.rotlib import load_library

    rout = Path(__file__).resolve().parent / "golden" / "1CRN.rotamers.pdb"
    if not rout.exists():
        pytest.skip("1CRN --rout golden missing")
    golden = _parse_rout(rout)
    lib = load_library(rotlib_dir)
    atoms = read_structure(structures_dir / "1CRN.pdb")
    positions = positions_from_atoms(atoms)

    compared = 0
    for pos in positions:
        for aa, heavy in SIDECHAIN_HEAVY.items():
            tmpl = lib.residues.get(aa)
            if tmpl is None:
                continue
            confs, weights = lib.rotamers_for(aa, pos.phi, pos.psi)
            if confs.shape[0] == 0:
                continue
            placed = place_rotamers(tmpl, pos.backbone, confs=confs, weights=weights)
            for r in range(confs.shape[0]):
                key = (pos.chain, pos.resnum, aa, r)
                if key not in golden:
                    continue
                g = golden[key]
                for atom in heavy:
                    idx = placed.atom_names.index(atom)
                    np.testing.assert_allclose(
                        placed.coords[r, idx], g[atom], atol=2e-3,
                        err_msg=f"{pos.chain},{pos.resnum} {aa} rot {r} {atom}",
                    )
                    compared += 1
    assert compared > 1000, f"only validated {compared} atom placements"
