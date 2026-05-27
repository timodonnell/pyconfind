"""Contact-degree calculation.

Given the surviving rotamers at each position, compute the contact degree
between every pair of positions and the "freedom" metric per position.

Mirrors ``confind.cpp::computeContactDegrees`` + ``contactProbability``:

* For each pair of positions ``(i, j)`` with CA-CA distance below the cutoff
  ``dcut`` (default 25 Å), iterate over rotamer pairs ``(ri, rj)``.
* Weight each pair by ``aaProp(ri) * rotProb(ri) * aaProp(rj) * rotProb(rj)``.
* Mark the pair as "in contact" iff any sidechain atom of ``ri`` is within
  ``contDist`` (default 3.0 Å) of any sidechain atom of ``rj``.
* Contact degree = (sum of weights where in-contact) / (sum of all weights).

The C++ uses PCA to sort positions and sweep along the principal axis. We use
``scipy.cKDTree`` on the CA coordinates for the same neighbor pruning — the
contact-degree value for each pair is invariant to iteration order.

For each contacting pair, we also accumulate per-rotamer collision-probability
mass (``collProbs``), used to compute the per-position "freedom" metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .build import PositionRotamers


@dataclass(frozen=True)
class Contact:
    """One pairwise residue contact with its degree."""

    pos_i: int
    pos_j: int
    degree: float


@dataclass(frozen=True)
class ContactReport:
    """Full per-pair + per-position contact report."""

    contacts: tuple[Contact, ...]
    # Per-position scalars, indexed by Position.index (==list index here).
    sum_contact_degree: np.ndarray   # (N,) float64
    freedom: np.ndarray               # (N,) float64
    # Per-position summary echoed back for convenience.
    crwdnes: np.ndarray               # (N,) float64
    permanent_contacts: tuple[frozenset[int], ...]


def compute_contacts(
    positions: list[PositionRotamers],
    *,
    dcut: float = 25.0,
    contact_dist: float = 3.0,
    freedom_type: int = 2,
) -> ContactReport:
    """Compute contact degree for every pair within ``dcut`` Å (CA-CA).

    ``freedom_type`` selects the per-position freedom formula from
    ``confind.cpp:797-826``. Type 2 (default) is the C++ default.
    """
    N = len(positions)
    ca = np.empty((N, 3), dtype=np.float64)
    for i, pr in enumerate(positions):
        ca[i] = pr.position.backbone.get("CA", np.full(3, np.nan))

    # Per-position sidechain stacks and per-rotamer (aaProp * rotProb) weights.
    stacked_xyz: list[np.ndarray] = []  # (R*A, 3) per position
    stacked_owner: list[np.ndarray] = []  # (R*A,) rotamer index per atom
    rot_weights: list[np.ndarray] = []     # (R,) aaProp * rotProb per rotamer
    rot_bbox_lo: list[np.ndarray] = []     # (R, 3) per-rotamer bbox
    rot_bbox_hi: list[np.ndarray] = []
    rot_counts: list[int] = []
    for pr in positions:
        if not pr.rotamers:
            stacked_xyz.append(np.zeros((0, 3)))
            stacked_owner.append(np.zeros(0, dtype=np.int64))
            rot_weights.append(np.zeros(0))
            rot_bbox_lo.append(np.zeros((0, 3)))
            rot_bbox_hi.append(np.zeros((0, 3)))
            rot_counts.append(0)
            continue
        pieces_xyz = []
        pieces_owner = []
        lo_list = []
        hi_list = []
        w_list = []
        for r, rot in enumerate(pr.rotamers):
            # For contact detection MSL uses heavy atoms only — sidechain H
            # atoms are stripped from the contact grid (see confind.cpp:650).
            heavy_mask = np.array(
                [not name.startswith("H") for name in rot.sidechain_atoms],
                dtype=bool,
            )
            xyz = rot.sidechain_xyz[heavy_mask]
            pieces_xyz.append(xyz)
            pieces_owner.append(np.full(xyz.shape[0], r, dtype=np.int64))
            if xyz.shape[0] == 0:
                lo_list.append(np.full(3, np.inf))
                hi_list.append(np.full(3, -np.inf))
            else:
                lo_list.append(xyz.min(axis=0))
                hi_list.append(xyz.max(axis=0))
            w_list.append(rot.aa_prop * rot.weight)
        stacked_xyz.append(np.vstack(pieces_xyz) if pieces_xyz else np.zeros((0, 3)))
        stacked_owner.append(
            np.concatenate(pieces_owner) if pieces_owner else np.zeros(0, dtype=np.int64)
        )
        rot_weights.append(np.asarray(w_list, dtype=np.float64))
        rot_bbox_lo.append(np.asarray(lo_list, dtype=np.float64))
        rot_bbox_hi.append(np.asarray(hi_list, dtype=np.float64))
        rot_counts.append(len(pr.rotamers))

    trees: list[cKDTree | None] = []
    for xyz in stacked_xyz:
        trees.append(cKDTree(xyz) if xyz.shape[0] > 0 else None)

    # Use a CA-distance prefilter (cKDTree on CA) — equivalent to the C++ PCA
    # sweep, but order-independent.
    valid = ~np.isnan(ca[:, 0])
    contacts: list[Contact] = []
    sum_cont = np.zeros(N, dtype=np.float64)
    collision_probs: list[np.ndarray] = [
        np.zeros(rc, dtype=np.float64) for rc in rot_counts
    ]

    if valid.any():
        valid_idx = np.flatnonzero(valid)
        ca_tree = cKDTree(ca[valid_idx])
        pairs = ca_tree.query_pairs(r=dcut, output_type="ndarray")
        # Convert back to absolute indices.
        for raw_i, raw_j in pairs:
            i = int(valid_idx[raw_i])
            j = int(valid_idx[raw_j])
            if i > j:
                i, j = j, i
            d = _pair_contact_degree(
                stacked_xyz[i], stacked_owner[i], trees[i], rot_bbox_lo[i], rot_bbox_hi[i],
                stacked_xyz[j], stacked_owner[j], trees[j], rot_bbox_lo[j], rot_bbox_hi[j],
                rot_weights[i], rot_weights[j],
                collision_probs[i], collision_probs[j],
                contact_dist,
            )
            if d > 0:
                contacts.append(Contact(pos_i=i, pos_j=j, degree=d))
                sum_cont[i] += d
                sum_cont[j] += d

    # Sort contacts the same way the C++ does (by (pos_i, pos_j) ascending).
    contacts.sort(key=lambda c: (c.pos_i, c.pos_j))

    # Per-position freedom (confind.cpp:797-826). When orig_num_rotamers is
    # zero (CA-only inputs, mostly), the C++ produces NaN via sqrt(0)/0 — we
    # match that to keep text output byte-identical.
    freedom = np.empty(N, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, pr in enumerate(positions):
            cp = collision_probs[i]
            orig = pr.num_rotamers_placed
            if freedom_type == 1:
                n = float((cp / 100.0 < 0.5).sum())
                freedom[i] = n / orig if orig != 0 else float("nan")
            elif freedom_type == 2:
                n1 = float((cp / 100.0 < 0.5).sum())
                n2 = float((cp / 100.0 < 2.0).sum())
                freedom[i] = (
                    float(np.sqrt((n1 * n1 + n2 * n2) / 2.0)) / orig
                    if orig != 0
                    else float("nan")
                )
            else:
                freedom[i] = 999.0

    crwdnes = np.array([pr.fraction_pruned for pr in positions], dtype=np.float64)
    perm = tuple(pr.permanent_contacts for pr in positions)
    return ContactReport(
        contacts=tuple(contacts),
        sum_contact_degree=sum_cont,
        freedom=freedom,
        crwdnes=crwdnes,
        permanent_contacts=perm,
    )


def _pair_contact_degree(
    xyz_i: np.ndarray,
    owner_i: np.ndarray,
    tree_i: cKDTree | None,
    bbox_lo_i: np.ndarray,
    bbox_hi_i: np.ndarray,
    xyz_j: np.ndarray,
    owner_j: np.ndarray,
    tree_j: cKDTree | None,
    bbox_lo_j: np.ndarray,
    bbox_hi_j: np.ndarray,
    weights_i: np.ndarray,
    weights_j: np.ndarray,
    coll_probs_i: np.ndarray,
    coll_probs_j: np.ndarray,
    contact_dist: float,
) -> float:
    """Return contact degree for one pair of positions; update collision probs in-place."""
    R_i = weights_i.size
    R_j = weights_j.size
    if R_i == 0 or R_j == 0 or tree_i is None or tree_j is None:
        return 0.0
    # Normalizer: sum over all (ri, rj) of p1*p2 = (sum p1) * (sum p2).
    sum_wi = float(weights_i.sum())
    sum_wj = float(weights_j.sum())
    n = sum_wi * sum_wj
    if n == 0.0:
        return 0.0
    # Bounding-box prefilter: rotamers of i and j that can possibly interact.
    # If position i's overall bbox is far from position j's overall bbox, no
    # contact possible (cheap reject).
    overall_lo_i = bbox_lo_i.min(axis=0)
    overall_hi_i = bbox_hi_i.max(axis=0)
    overall_lo_j = bbox_lo_j.min(axis=0)
    overall_hi_j = bbox_hi_j.max(axis=0)
    if np.any(overall_lo_i > overall_hi_j + contact_dist) or np.any(
        overall_lo_j > overall_hi_i + contact_dist
    ):
        return 0.0
    # For each atom of i, find atoms of j within contact_dist. Aggregate by
    # (i_owner, j_owner) to get the set of contacting rotamer pairs.
    neighbors = tree_j.query_ball_point(xyz_i, r=contact_dist)
    # Build a sparse boolean matrix of in-contact rotamer pairs.
    contact_pairs: set[tuple[int, int]] = set()
    for ai, js in enumerate(neighbors):
        if not js:
            continue
        ri = int(owner_i[ai])
        for aj in js:
            rj = int(owner_j[aj])
            contact_pairs.add((ri, rj))
    if not contact_pairs:
        return 0.0
    # Accumulate contact mass + collision probabilities.
    c = 0.0
    for ri, rj in contact_pairs:
        pij = float(weights_i[ri] * weights_j[rj])
        c += pij
        coll_probs_i[ri] += float(weights_j[rj])
        coll_probs_j[rj] += float(weights_i[ri])
    return c / n
