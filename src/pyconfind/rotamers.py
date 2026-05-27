"""Place rotamers onto a target backbone using a rotamer library's IC table.

The C++ confind builds rotamers by:

1. Taking the native N, CA, C, O backbone of a residue position.
2. For each amino-acid identity being considered, looking up that AA's IC
   template (from EBL.out) and placing each mobile atom in the order given by
   the DEFI lines.
3. Each mobile atom is placed against three already-placed atoms (some
   combination of backbone + previously-placed mobiles).

We follow the same procedure, vectorized over rotamers: for each placement
step, we build ``(R, 3)`` parent arrays where ``R`` is the number of rotamers
and call :func:`pyconfind.geometry.place_batch` once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import place_batch
from .rotlib import ResidueICTemplate


@dataclass(frozen=True)
class PlacedRotamers:
    """All rotamer placements for one (position, identity).

    ``atom_names`` is the ordered list of atoms in the rotamer (backbone first
    in input order, then sidechain in placement order). ``coords`` has shape
    ``(R, A, 3)``. ``weights`` has shape ``(R,)``. ``placed_mask`` is ``(A,)``
    of bool indicating which atoms were placed via IC (vs. copied from
    backbone).
    """

    atom_names: tuple[str, ...]
    coords: np.ndarray  # (R, A, 3)
    weights: np.ndarray  # (R,)
    placed_mask: np.ndarray  # (A,) bool


def place_rotamers(
    template: ResidueICTemplate,
    backbone_coords: dict[str, np.ndarray],
    weights: np.ndarray | None = None,
    confs: np.ndarray | None = None,
) -> PlacedRotamers:
    """Place every rotamer of ``template`` on ``backbone_coords``.

    Parameters
    ----------
    template
        IC template returned by :func:`pyconfind.rotlib.parse_ebl`.
    backbone_coords
        Maps atom name to ``(3,)`` coordinate. Must contain every parent atom
        not in the placed set. For typical AAs, ``N``, ``CA``, ``C`` are
        required; the C++ also copies ``O`` if present.
    weights, confs
        Override ``template.weights`` and ``template.confs`` to restrict to a
        subset (e.g. only the rotamers active for the current phi/psi bin).
        If ``None``, uses the full pool from the template.
    """
    if confs is None:
        confs = template.confs
    if weights is None:
        weights = template.weights
    placed_names = template.placed
    parent_names = template.parents
    R, M, _ = confs.shape

    # Atom layout: backbone atoms (in dict-iteration order) + placed atoms (in
    # template.placed order). We need to know the index of every parent atom
    # to gather coords during the loop.
    bb_names = list(backbone_coords.keys())
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(bb_names)}
    for k, name in enumerate(placed_names):
        name_to_idx[name] = len(bb_names) + k

    A = len(bb_names) + len(placed_names)
    coords = np.empty((R, A, 3), dtype=np.float64)
    # Broadcast backbone coords across all rotamers.
    for n in bb_names:
        coords[:, name_to_idx[n], :] = backbone_coords[n]

    # Place each mobile atom in template order.
    for k in range(M):
        a_name, b_name, c_name = parent_names[k]
        a = coords[:, name_to_idx[str(a_name)], :]
        b = coords[:, name_to_idx[str(b_name)], :]
        c = coords[:, name_to_idx[str(c_name)], :]
        d = place_batch(
            a,
            b,
            c,
            bond=confs[:, k, 2],
            angle_deg=confs[:, k, 1],
            dihedral_deg=confs[:, k, 0],
        )
        coords[:, name_to_idx[placed_names[k]], :] = d

    atom_names = tuple(bb_names) + placed_names
    placed_mask = np.zeros(A, dtype=bool)
    placed_mask[len(bb_names):] = True
    return PlacedRotamers(
        atom_names=atom_names,
        coords=coords,
        weights=np.asarray(weights, dtype=np.float64),
        placed_mask=placed_mask,
    )
