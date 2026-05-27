"""Per-position structure helpers: backbone access, phi/psi computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pdb import Atoms, position_iter


@dataclass(frozen=True)
class Position:
    """One residue position with its native backbone and (optional) phi/psi."""

    index: int
    chain: str
    resnum: int
    icode: str
    resname: str
    backbone: dict[str, np.ndarray]   # name -> (3,) coord
    phi: float | None  # degrees, None if undefined (terminus / missing atoms)
    psi: float | None
    omega: float | None


_BACKBONE_NAMES = ("N", "CA", "C", "O")


def positions_from_atoms(atoms: Atoms) -> list[Position]:
    """Build :class:`Position` records from parsed PDB atoms.

    For positions with multiple identities, the *first* identity in input order
    is treated as the native one (matches MSL's ``getCurrentIdentity()``).
    """
    slices = position_iter(atoms)
    # First pass: for each position, find the native identity's backbone atoms.
    natives: list[dict[str, np.ndarray]] = []
    headers: list[tuple[int, str, int, str, str]] = []
    for pi, s in enumerate(slices):
        identity = atoms.identity_index[s]
        native_mask = identity == 0
        sub_names = atoms.name[s][native_mask]
        sub_xyz = atoms.xyz[s][native_mask]
        bb: dict[str, np.ndarray] = {}
        for n, xyz in zip(sub_names, sub_xyz, strict=True):
            if n in _BACKBONE_NAMES:
                bb[n] = xyz
        first = s.start
        # Pick resname from the first native atom.
        first_native = int(np.flatnonzero(native_mask)[0]) + s.start
        headers.append(
            (
                pi,
                str(atoms.chain[first_native]),
                int(atoms.resnum[first_native]),
                str(atoms.icode[first_native]),
                str(atoms.resname[first_native]),
            )
        )
        natives.append(bb)
        _ = first

    # Second pass: compute phi/psi/omega using neighbors within the same chain.
    positions: list[Position] = []
    for pi, (idx, chain, resnum, icode, resname) in enumerate(headers):
        prev_bb = natives[pi - 1] if pi > 0 and headers[pi - 1][1] == chain else None
        next_bb = (
            natives[pi + 1]
            if pi + 1 < len(headers) and headers[pi + 1][1] == chain
            else None
        )
        bb = natives[pi]
        phi = _safe_dihedral(prev_bb, "C", bb, "N", bb, "CA", bb, "C") if prev_bb else None
        psi = _safe_dihedral(bb, "N", bb, "CA", bb, "C", next_bb, "N") if next_bb else None
        # MSL's omega for the current position is CA(prev) - C(prev) - N - CA.
        omega = (
            _safe_dihedral(prev_bb, "CA", prev_bb, "C", bb, "N", bb, "CA")
            if prev_bb
            else None
        )
        positions.append(
            Position(
                index=idx,
                chain=chain,
                resnum=resnum,
                icode=icode,
                resname=resname,
                backbone=bb,
                phi=phi,
                psi=psi,
                omega=omega,
            )
        )
    return positions


def _safe_dihedral(
    src_a: dict[str, np.ndarray] | None,
    name_a: str,
    src_b: dict[str, np.ndarray] | None,
    name_b: str,
    src_c: dict[str, np.ndarray] | None,
    name_c: str,
    src_d: dict[str, np.ndarray] | None,
    name_d: str,
) -> float | None:
    """Return the dihedral A-B-C-D in degrees, or ``None`` if any atom missing."""
    if src_a is None or src_b is None or src_c is None or src_d is None:
        return None
    try:
        a = src_a[name_a]
        b = src_b[name_b]
        c = src_c[name_c]
        d = src_d[name_d]
    except KeyError:
        return None
    return float(dihedral_deg(a, b, c, d))


def dihedral_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """IUPAC dihedral angle A-B-C-D in degrees, in [-180, 180].

    Sign convention: positive when D rotates clockwise viewed from B → C
    (i.e. looking along the B→C axis). Matches MSL's ``Atom::getDihedral``.
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n2 = np.cross(b2, b3)
    y = float(np.linalg.norm(b2)) * float(np.dot(b1, n2))
    x = float(np.dot(np.cross(b1, b2), n2))
    return float(np.degrees(np.arctan2(y, x)))
