"""Numba-accelerated contact-degree backend.

A drop-in fast path for :func:`pyconfind.contacts.compute_contacts`. The pure
NumPy/SciPy implementation in ``contacts.py`` remains the reference; this
module reproduces it with a JIT-compiled, multi-threaded kernel.

Design
------
All heavy sidechain atoms are flattened into contiguous global arrays grouped
by position (CSR layout). The kernel parallelizes over *positions*: each
thread fully owns the collision-probability row of its outer position, so
there are no write races and no need for thread-id hacks or atomic adds. The
cost is that each position pair's contact set is computed from both ends
(once per direction); in exchange the parallelism is trivially correct.

Within a position pair, contacts are found by brute force over atom pairs with
a per-position bounding-box reject and a generation-counter scratch array that
deduplicates rotamer pairs without re-zeroing.

Floating-point summation order differs from the reference, but the resulting
error (~1e-11) is far below the 1e-6 print precision, so output stays
byte-identical.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from .build import PositionRotamers
from .contacts import Contact, ContactReport


def _flatten(positions: list[PositionRotamers]) -> tuple:
    """Pack positions into CSR-style global arrays for the kernel."""
    P = len(positions)
    pos_atom_start = np.zeros(P + 1, dtype=np.int64)
    pos_rot_start = np.zeros(P + 1, dtype=np.int64)
    xyz_chunks: list[np.ndarray] = []
    rot_local_chunks: list[np.ndarray] = []
    weights: list[float] = []
    ca = np.full((P, 3), np.nan, dtype=np.float64)
    bbox_lo = np.full((P, 3), np.inf, dtype=np.float64)
    bbox_hi = np.full((P, 3), -np.inf, dtype=np.float64)
    nrot = np.zeros(P, dtype=np.int64)

    # All rotamers of the same AA at a position share the same sidechain_atoms
    # tuple object, so cache the heavy-atom mask by identity.
    heavy_cache: dict[int, np.ndarray] = {}

    def heavy_mask(names: tuple[str, ...]) -> np.ndarray:
        key = id(names)
        m = heavy_cache.get(key)
        if m is None:
            m = np.array([not n.startswith("H") for n in names], dtype=bool)
            heavy_cache[key] = m
        return m

    for i, pr in enumerate(positions):
        ca[i] = pr.position.backbone.get("CA", np.full(3, np.nan))
        nrot[i] = len(pr.rotamers)
        a_count = 0
        for r, rot in enumerate(pr.rotamers):
            heavy = heavy_mask(rot.sidechain_atoms)
            x = rot.sidechain_xyz[heavy]
            if x.shape[0]:
                xyz_chunks.append(x)
                rot_local_chunks.append(np.full(x.shape[0], r, dtype=np.int32))
                bbox_lo[i] = np.minimum(bbox_lo[i], x.min(axis=0))
                bbox_hi[i] = np.maximum(bbox_hi[i], x.max(axis=0))
                a_count += x.shape[0]
            weights.append(rot.aa_prop * rot.weight)
        pos_atom_start[i + 1] = pos_atom_start[i] + a_count
        pos_rot_start[i + 1] = pos_rot_start[i] + len(pr.rotamers)

    xyz = np.vstack(xyz_chunks) if xyz_chunks else np.zeros((0, 3))
    rot_local = (
        np.concatenate(rot_local_chunks) if rot_local_chunks else np.zeros(0, np.int32)
    )
    rot_w = np.asarray(weights, dtype=np.float64) if weights else np.zeros(0)
    pos_sum_w = np.zeros(P, dtype=np.float64)
    for i in range(P):
        s, e = pos_rot_start[i], pos_rot_start[i + 1]
        pos_sum_w[i] = float(rot_w[s:e].sum())
    return (
        xyz, rot_local, pos_atom_start, pos_rot_start, rot_w, pos_sum_w,
        ca, bbox_lo, bbox_hi, nrot,
    )


def _build_adjacency(ca: np.ndarray, dcut: float) -> tuple:
    """For each position, the list of partner positions within ``dcut`` (CA-CA).

    Returns CSR arrays ``(adj_start, adj_partner, adj_slot)`` covering *both*
    directions of every pair, plus ``pair_i/pair_j`` for the unique i<j pairs.
    ``adj_slot[e]`` is the unique pair slot when ``partner > i`` else -1.
    """
    P = ca.shape[0]
    valid = ~np.isnan(ca[:, 0])
    if not valid.any():
        return (
            np.zeros(P + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64),
            np.zeros(0, np.int64), np.zeros(0, np.int64),
        )
    vidx = np.flatnonzero(valid)
    tree = cKDTree(ca[vidx])
    raw = tree.query_pairs(r=dcut, output_type="ndarray")
    pair_i = np.minimum(vidx[raw[:, 0]], vidx[raw[:, 1]]).astype(np.int64)
    pair_j = np.maximum(vidx[raw[:, 0]], vidx[raw[:, 1]]).astype(np.int64)
    npairs = pair_i.shape[0]
    # Build both-direction adjacency with degree counting.
    deg = np.zeros(P, dtype=np.int64)
    np.add.at(deg, pair_i, 1)
    np.add.at(deg, pair_j, 1)
    adj_start = np.zeros(P + 1, dtype=np.int64)
    adj_start[1:] = np.cumsum(deg)
    adj_partner = np.empty(adj_start[-1], dtype=np.int64)
    adj_slot = np.empty(adj_start[-1], dtype=np.int64)
    cursor = adj_start[:-1].copy()
    for s in range(npairs):
        i, j = int(pair_i[s]), int(pair_j[s])
        # i<j direction carries the unique pair slot; j>i direction carries -1.
        e = cursor[i]
        adj_partner[e] = j
        adj_slot[e] = s
        cursor[i] += 1
        e = cursor[j]
        adj_partner[e] = i
        adj_slot[e] = -1
        cursor[j] += 1
    return adj_start, adj_partner, adj_slot, pair_i, pair_j


@njit(parallel=True, cache=True, fastmath=False)
def _contact_kernel(  # type: ignore[no-untyped-def]
    xyz, rot_local, pos_atom_start, pos_rot_start, rot_w, pos_sum_w,
    bbox_lo, bbox_hi, nrot, adj_start, adj_partner, adj_slot,
    npairs, max_rot, contact_dist,
):
    P = nrot.shape[0]
    cut2 = contact_dist * contact_dist
    cp = np.zeros(rot_w.shape[0], dtype=np.float64)
    degree = np.full(npairs, np.nan, dtype=np.float64)
    for i in prange(P):
        Ri = nrot[i]
        if Ri == 0:
            continue
        ai0 = pos_atom_start[i]
        ai1 = pos_atom_start[i + 1]
        wi0 = pos_rot_start[i]
        seen = np.full(Ri * max_rot, -1, dtype=np.int64)
        gen = 0
        for e in range(adj_start[i], adj_start[i + 1]):
            j = adj_partner[e]
            slot = adj_slot[e]
            Rj = nrot[j]
            if Rj == 0:
                continue
            # Bounding-box reject (matches the reference prefilter).
            far = False
            for d in range(3):
                if bbox_lo[i, d] > bbox_hi[j, d] + contact_dist:
                    far = True
                if bbox_lo[j, d] > bbox_hi[i, d] + contact_dist:
                    far = True
            if far:
                continue
            gen += 1
            aj0 = pos_atom_start[j]
            aj1 = pos_atom_start[j + 1]
            wj0 = pos_rot_start[j]
            c = 0.0
            record = slot >= 0
            for a in range(ai0, ai1):
                ra = rot_local[a]
                xa = xyz[a, 0]
                ya = xyz[a, 1]
                za = xyz[a, 2]
                base = ra * max_rot
                for b in range(aj0, aj1):
                    dx = xa - xyz[b, 0]
                    dy = ya - xyz[b, 1]
                    dz = za - xyz[b, 2]
                    if dx * dx + dy * dy + dz * dz <= cut2:
                        rb = rot_local[b]
                        idx = base + rb
                        if seen[idx] != gen:
                            seen[idx] = gen
                            wrb = rot_w[wj0 + rb]
                            cp[wi0 + ra] += wrb
                            if record:
                                c += rot_w[wi0 + ra] * wrb
            if record:
                n = pos_sum_w[i] * pos_sum_w[j]
                if n != 0.0:
                    degree[slot] = c / n
                # n == 0 leaves degree[slot] = nan (set at init)
    return cp, degree


def compute_contacts_numba(
    positions: list[PositionRotamers],
    *,
    dcut: float = 25.0,
    contact_dist: float = 3.0,
    freedom_type: int = 2,
) -> ContactReport:
    """Numba-accelerated equivalent of :func:`pyconfind.contacts.compute_contacts`."""
    P = len(positions)
    (
        xyz, rot_local, pos_atom_start, pos_rot_start, rot_w, pos_sum_w,
        ca, bbox_lo, bbox_hi, nrot,
    ) = _flatten(positions)
    adj_start, adj_partner, adj_slot, pair_i, pair_j = _build_adjacency(ca, dcut)
    npairs = pair_i.shape[0]
    max_rot = int(nrot.max()) if P and nrot.max() > 0 else 1

    cp_flat, degree = _contact_kernel(
        xyz, rot_local, pos_atom_start, pos_rot_start, rot_w, pos_sum_w,
        bbox_lo, bbox_hi, nrot, adj_start, adj_partner, adj_slot,
        npairs, max_rot, float(contact_dist),
    )

    # Assemble contacts (degree > 0), sorted by (pos_i, pos_j).
    contacts: list[Contact] = []
    sum_cont = np.zeros(P, dtype=np.float64)
    pos = np.lexsort((pair_j, pair_i))
    for s in pos:
        d = degree[s]
        if d > 0:
            i, j = int(pair_i[s]), int(pair_j[s])
            contacts.append(Contact(pos_i=i, pos_j=j, degree=float(d)))
            sum_cont[i] += d
            sum_cont[j] += d

    # Per-position freedom from collision probabilities (matches reference).
    freedom = np.empty(P, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, pr in enumerate(positions):
            s, e = pos_rot_start[i], pos_rot_start[i + 1]
            cp = cp_flat[s:e]
            orig = pr.num_rotamers_placed
            if freedom_type == 1:
                n = float((cp / 100.0 < 0.5).sum())
                freedom[i] = n / orig if orig != 0 else float("nan")
            elif freedom_type == 2:
                n1 = float((cp / 100.0 < 0.5).sum())
                n2 = float((cp / 100.0 < 2.0).sum())
                freedom[i] = (
                    float(np.sqrt((n1 * n1 + n2 * n2) / 2.0)) / orig
                    if orig != 0 else float("nan")
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
