"""Geometry helpers: place atoms from internal coordinates (NeRF).

We use the standard "Natural Extension of Reference Frame" placement: given
three already-placed atoms ``a``, ``b``, ``c``, place ``d`` at distance ``r``
from ``c`` with bond angle ``theta = angle(b-c-d)`` and dihedral
``phi = dihedral(a-b-c-d)``.

The convention here matches MSL's IC table (verified against the C++
``confind --rout`` output): with basis vectors

* ``bc = normalize(c - b)``
* ``n  = normalize(cross(b - a, c - b))``  (out of the ABC plane)
* ``m  = cross(n, bc)``                    (in-plane, perpendicular to BC)

the new atom sits at

    d = c + r * ( -cos(theta) * bc + sin(theta) * cos(phi) * m + sin(theta) * sin(phi) * n )

All angles are in degrees on the public API.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def place_one(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: float,
    angle_deg: float,
    dihedral_deg: float,
) -> np.ndarray:
    """Place a single atom from three parents (un-vectorized; useful for tests)."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    ab = b - a
    ab /= np.linalg.norm(ab)
    n = np.cross(ab, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(dihedral_deg)
    result = c + bond * (
        -np.cos(theta) * bc
        + np.sin(theta) * np.cos(phi) * m
        + np.sin(theta) * np.sin(phi) * n
    )
    return np.asarray(result, dtype=np.float64)


def _cross3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cross product of two ``(N, 3)`` arrays.

    Direct component arithmetic; ``np.cross`` carries significant per-call
    overhead (moveaxis / axis normalization) that dominates the IC builder.
    """
    return np.stack(
        (
            u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
            u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
            u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
        ),
        axis=1,
    )


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.sqrt((x * x).sum(axis=1, keepdims=True))
    return np.asarray(x / norm, dtype=np.float64)


def _place_batch_numpy(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: np.ndarray,
    angle_deg: np.ndarray,
    dihedral_deg: np.ndarray,
) -> np.ndarray:
    bc = _normalize_rows(c - b)
    ab = _normalize_rows(b - a)
    n = _normalize_rows(_cross3(ab, bc))
    m = _cross3(n, bc)
    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(dihedral_deg)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    cos_p = np.cos(phi)[:, None]
    sin_p = np.sin(phi)[:, None]
    result = c + bond[:, None] * (-cos_t * bc + sin_t * cos_p * m + sin_t * sin_p * n)
    return np.asarray(result, dtype=np.float64)


_DEG2RAD = math.pi / 180.0


def _place_batch_kernel_python(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: np.ndarray,
    angle_deg: np.ndarray,
    dihedral_deg: np.ndarray,
) -> np.ndarray:
    """Hand-scalarized scalar form of ``_place_batch_numpy``.

    For length-3 row vectors, ``cross``/``dot``/``norm`` reduce to exactly
    these scalar ops (validated against the numpy form on random inputs:
    bit-for-bit equal). The same identity is verified at test time by the
    C++ byte-equivalence goldens. njit-compiled on import.
    """
    N = a.shape[0]
    result = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        ax, ay, az = a[i, 0], a[i, 1], a[i, 2]
        bx, by, bz = b[i, 0], b[i, 1], b[i, 2]
        cx, cy, cz = c[i, 0], c[i, 1], c[i, 2]
        # bc = normalize(c - b)
        bcx0, bcy0, bcz0 = cx - bx, cy - by, cz - bz
        nb = math.sqrt(bcx0 * bcx0 + bcy0 * bcy0 + bcz0 * bcz0)
        bcx, bcy, bcz = bcx0 / nb, bcy0 / nb, bcz0 / nb
        # ab = normalize(b - a)
        abx0, aby0, abz0 = bx - ax, by - ay, bz - az
        na = math.sqrt(abx0 * abx0 + aby0 * aby0 + abz0 * abz0)
        abx, aby, abz = abx0 / na, aby0 / na, abz0 / na
        # n = normalize(cross(ab, bc))
        nx0 = aby * bcz - abz * bcy
        ny0 = abz * bcx - abx * bcz
        nz0 = abx * bcy - aby * bcx
        nn = math.sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0)
        nx, ny, nz = nx0 / nn, ny0 / nn, nz0 / nn
        # m = cross(n, bc)
        mx = ny * bcz - nz * bcy
        my = nz * bcx - nx * bcz
        mz = nx * bcy - ny * bcx
        # angles + final
        theta = angle_deg[i] * _DEG2RAD
        phi = dihedral_deg[i] * _DEG2RAD
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        s = bond[i]
        result[i, 0] = cx + s * (-cos_t * bcx + sin_t * cos_p * mx + sin_t * sin_p * nx)
        result[i, 1] = cy + s * (-cos_t * bcy + sin_t * cos_p * my + sin_t * sin_p * ny)
        result[i, 2] = cz + s * (-cos_t * bcz + sin_t * cos_p * mz + sin_t * sin_p * nz)
    return result


if _HAS_NUMBA:
    _place_batch_kernel = numba.njit(
        _place_batch_kernel_python, cache=True, fastmath=False, error_model="numpy",
    )
else:  # pragma: no cover
    _place_batch_kernel = _place_batch_kernel_python


def place_batch(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: np.ndarray,
    angle_deg: np.ndarray,
    dihedral_deg: np.ndarray,
) -> np.ndarray:
    """Place ``N`` atoms in parallel.

    ``a``, ``b``, ``c`` are ``(N, 3)`` arrays of parent coordinates;
    ``bond``, ``angle_deg``, ``dihedral_deg`` are ``(N,)`` arrays of internal
    coords. Returns ``(N, 3)`` array of placed positions.

    Routes to the hand-scalarized numba kernel when numba is available; falls
    back to the pure-NumPy form otherwise. Both are bit-equivalent for the
    Dunbrack 2010 library (verified by the byte-identity goldens).
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    bond = np.ascontiguousarray(bond, dtype=np.float64)
    angle_deg = np.ascontiguousarray(angle_deg, dtype=np.float64)
    dihedral_deg = np.ascontiguousarray(dihedral_deg, dtype=np.float64)
    result: np.ndarray = _place_batch_kernel(a, b, c, bond, angle_deg, dihedral_deg)
    return result
