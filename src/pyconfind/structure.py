"""Per-position structure helpers: backbone access, phi/psi computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pdb import Atoms, position_iter

try:
    import numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when extras aren't installed
    _HAS_NUMBA = False


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


def _dihedrals_kernel_python(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray,
    chain_break: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute phi/psi/omega for every position as ``(P,)`` arrays in degrees.

    Inputs are ``(P, 3)`` arrays of N/CA/C coords (NaN where an atom is
    missing) plus a ``(P,)`` boolean ``chain_break`` flagging the first
    position of each chain. Output is ``NaN`` where the dihedral is
    undefined (terminus, missing atoms, or chain break).

    The dihedral arithmetic is hand-scalarized over the length-3 vectors
    but byte-for-byte equivalent to the original ``dihedral_deg``: for
    length-3 vectors ``np.dot``/``np.cross``/``np.linalg.norm`` reduce to
    exactly these scalar ops (verified at test time against the goldens).
    This function is njit-compiled on import; the Python form below is
    only used if ``numba`` is not installed.
    """
    P = n_xyz.shape[0]
    phi = np.full(P, np.nan)
    psi = np.full(P, np.nan)
    omega = np.full(P, np.nan)
    for i in range(P):
        if math.isnan(n_xyz[i, 0]) or math.isnan(ca_xyz[i, 0]) or math.isnan(c_xyz[i, 0]):
            continue
        has_prev = i > 0 and not chain_break[i] and not math.isnan(c_xyz[i - 1, 0])
        has_next = (
            i + 1 < P and not chain_break[i + 1] and not math.isnan(n_xyz[i + 1, 0])
        )
        if has_prev:
            # phi: C(prev) - N(i) - CA(i) - C(i)
            ax, ay, az = c_xyz[i - 1, 0], c_xyz[i - 1, 1], c_xyz[i - 1, 2]
            bx, by, bz = n_xyz[i, 0], n_xyz[i, 1], n_xyz[i, 2]
            cx_, cy_, cz_ = ca_xyz[i, 0], ca_xyz[i, 1], ca_xyz[i, 2]
            dx, dy, dz = c_xyz[i, 0], c_xyz[i, 1], c_xyz[i, 2]
            b1x, b1y, b1z = bx - ax, by - ay, bz - az
            b2x, b2y, b2z = cx_ - bx, cy_ - by, cz_ - bz
            b3x, b3y, b3z = dx - cx_, dy - cy_, dz - cz_
            n2x = b2y * b3z - b2z * b3y
            n2y = b2z * b3x - b2x * b3z
            n2z = b2x * b3y - b2y * b3x
            crx = b1y * b2z - b1z * b2y
            cry = b1z * b2x - b1x * b2z
            crz = b1x * b2y - b1y * b2x
            norm_b2 = math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
            yv = norm_b2 * (b1x * n2x + b1y * n2y + b1z * n2z)
            xv = crx * n2x + cry * n2y + crz * n2z
            phi[i] = math.degrees(math.atan2(yv, xv))

            if not math.isnan(ca_xyz[i - 1, 0]):
                # omega: CA(prev) - C(prev) - N(i) - CA(i)
                ax, ay, az = ca_xyz[i - 1, 0], ca_xyz[i - 1, 1], ca_xyz[i - 1, 2]
                bx, by, bz = c_xyz[i - 1, 0], c_xyz[i - 1, 1], c_xyz[i - 1, 2]
                cx_, cy_, cz_ = n_xyz[i, 0], n_xyz[i, 1], n_xyz[i, 2]
                dx, dy, dz = ca_xyz[i, 0], ca_xyz[i, 1], ca_xyz[i, 2]
                b1x, b1y, b1z = bx - ax, by - ay, bz - az
                b2x, b2y, b2z = cx_ - bx, cy_ - by, cz_ - bz
                b3x, b3y, b3z = dx - cx_, dy - cy_, dz - cz_
                n2x = b2y * b3z - b2z * b3y
                n2y = b2z * b3x - b2x * b3z
                n2z = b2x * b3y - b2y * b3x
                crx = b1y * b2z - b1z * b2y
                cry = b1z * b2x - b1x * b2z
                crz = b1x * b2y - b1y * b2x
                norm_b2 = math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
                yv = norm_b2 * (b1x * n2x + b1y * n2y + b1z * n2z)
                xv = crx * n2x + cry * n2y + crz * n2z
                omega[i] = math.degrees(math.atan2(yv, xv))

        if has_next:
            # psi: N(i) - CA(i) - C(i) - N(next)
            ax, ay, az = n_xyz[i, 0], n_xyz[i, 1], n_xyz[i, 2]
            bx, by, bz = ca_xyz[i, 0], ca_xyz[i, 1], ca_xyz[i, 2]
            cx_, cy_, cz_ = c_xyz[i, 0], c_xyz[i, 1], c_xyz[i, 2]
            dx, dy, dz = n_xyz[i + 1, 0], n_xyz[i + 1, 1], n_xyz[i + 1, 2]
            b1x, b1y, b1z = bx - ax, by - ay, bz - az
            b2x, b2y, b2z = cx_ - bx, cy_ - by, cz_ - bz
            b3x, b3y, b3z = dx - cx_, dy - cy_, dz - cz_
            n2x = b2y * b3z - b2z * b3y
            n2y = b2z * b3x - b2x * b3z
            n2z = b2x * b3y - b2y * b3x
            crx = b1y * b2z - b1z * b2y
            cry = b1z * b2x - b1x * b2z
            crz = b1x * b2y - b1y * b2x
            norm_b2 = math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
            yv = norm_b2 * (b1x * n2x + b1y * n2y + b1z * n2z)
            xv = crx * n2x + cry * n2y + crz * n2z
            psi[i] = math.degrees(math.atan2(yv, xv))
    return phi, psi, omega


if _HAS_NUMBA:
    _dihedrals_kernel = numba.njit(_dihedrals_kernel_python, cache=True)
else:  # pragma: no cover
    _dihedrals_kernel = _dihedrals_kernel_python


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

    # Second pass: pack backbone N/CA/C into (P, 3) arrays (NaN where absent)
    # and run the batched phi/psi/omega kernel. This replaces what used to be
    # a per-position Python loop of np.cross/np.dot calls on (3,) vectors —
    # the dominant cost of positions_from_atoms on large structures.
    P = len(headers)
    nan3 = np.full(3, np.nan)
    n_xyz = np.empty((P, 3))
    ca_xyz = np.empty((P, 3))
    c_xyz = np.empty((P, 3))
    chain_break = np.zeros(P, dtype=np.bool_)
    prev_chain = ""
    for pi, header in enumerate(headers):
        chain = header[1]
        bb = natives[pi]
        n_xyz[pi] = bb.get("N", nan3)
        ca_xyz[pi] = bb.get("CA", nan3)
        c_xyz[pi] = bb.get("C", nan3)
        chain_break[pi] = pi == 0 or chain != prev_chain
        prev_chain = chain

    phi_arr, psi_arr, omega_arr = _dihedrals_kernel(n_xyz, ca_xyz, c_xyz, chain_break)

    positions: list[Position] = []
    for pi, (idx, chain, resnum, icode, resname) in enumerate(headers):
        phi_v = float(phi_arr[pi])
        psi_v = float(psi_arr[pi])
        omega_v = float(omega_arr[pi])
        positions.append(
            Position(
                index=idx,
                chain=chain,
                resnum=resnum,
                icode=icode,
                resname=resname,
                backbone=natives[pi],
                phi=None if math.isnan(phi_v) else phi_v,
                psi=None if math.isnan(psi_v) else psi_v,
                omega=None if math.isnan(omega_v) else omega_v,
            )
        )
    return positions


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
