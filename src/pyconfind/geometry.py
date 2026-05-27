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

import numpy as np


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
    return (
        c
        + bond
        * (
            -np.cos(theta) * bc
            + np.sin(theta) * np.cos(phi) * m
            + np.sin(theta) * np.sin(phi) * n
        )
    )


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
    """
    bc = c - b
    bc /= np.linalg.norm(bc, axis=1, keepdims=True)
    ab = b - a
    ab /= np.linalg.norm(ab, axis=1, keepdims=True)
    n = np.cross(ab, bc)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    m = np.cross(n, bc)
    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(dihedral_deg)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    cos_p = np.cos(phi)[:, None]
    sin_p = np.sin(phi)[:, None]
    return c + bond[:, None] * (-cos_t * bc + sin_t * cos_p * m + sin_t * sin_p * n)
