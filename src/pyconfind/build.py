"""End-to-end rotamer building + backbone-clash pruning.

This is the Python equivalent of ``confind.cpp::filterRotamers``: for each
position and each amino-acid identity, place all rotamers and prune those
that clash with the backbone of *other* positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .rotamers import place_rotamers
from .rotlib import ResidueICTemplate, RotamerLibrary
from .structure import Position

# AAs to consider at each position. Order matches confind.cpp:457.
_DEFAULT_AAS: tuple[str, ...] = (
    "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL", "ALA",
)
# Atoms considered "backbone" by ``isBackbone`` in confind.cpp:1050. We only
# put the heavy ones in the clash tree, but the test must skip *any* of these
# as a sidechain atom.
_BACKBONE_NAMES = frozenset({"N", "C", "CA", "H", "O", "NT", "HA", "HN"})
# Heavy backbone atoms only: confind.cpp:523 excludes H-prefixed names from
# the global backbone tree.
_HEAVY_BACKBONE_PREFIX = "H"

# Amino acid propensities (% of residues in PDB), confind.cpp:235.
DEFAULT_AA_PROPENSITY: dict[str, float] = {
    "ALA": 7.73, "CYS": 1.84, "ASP": 5.82, "GLU": 6.61, "PHE": 4.05,
    "GLY": 7.11, "HIS": 2.35, "HSD": 2.35, "ILE": 5.66, "LYS": 6.27,
    "LEU": 8.83, "MET": 2.08, "ASN": 4.50, "PRO": 4.52, "GLN": 3.94,
    "ARG": 5.03, "SER": 6.13, "THR": 5.53, "VAL": 6.91, "TRP": 1.51,
    "TYR": 3.54,
}


@dataclass(frozen=True)
class SurvivingRotamer:
    """One surviving rotamer at one (position, AA identity)."""

    aa: str
    rot_id: int                    # zero-indexed within the AA pool that was placed
    weight: float                  # rotamer weight from the library
    aa_prop: float                 # amino-acid propensity (%)
    sidechain_atoms: tuple[str, ...]
    sidechain_xyz: np.ndarray      # (Asc, 3)
    backbone_atoms: tuple[str, ...]
    backbone_xyz: np.ndarray       # (Abb, 3)


@dataclass(frozen=True)
class PositionRotamers:
    """All surviving rotamers at one position, plus bookkeeping."""

    position: Position
    rotamers: tuple[SurvivingRotamer, ...]
    permanent_contacts: frozenset[int]  # other position indices in unavoidable contact
    fraction_pruned: float
    num_rotamers_placed: int  # total rotamers placed *before* pruning


def build_clash_tree(positions: list[Position]) -> tuple[cKDTree, np.ndarray]:
    """Build a single KD tree over heavy backbone atoms across all positions.

    Returns ``(tree, owner)`` where ``owner[k]`` is the position index that
    owns backbone atom ``k`` in the tree. Atoms with names starting with ``H``
    are excluded, matching the C++ filter on the global ``bbNN`` index.
    """
    pts: list[np.ndarray] = []
    owner: list[int] = []
    for p in positions:
        for name, xyz in p.backbone.items():
            if name.startswith(_HEAVY_BACKBONE_PREFIX):
                continue
            pts.append(xyz)
            owner.append(p.index)
    if not pts:
        # cKDTree needs at least one point; build an empty tree by passing
        # a single far-away point owned by -1 so all queries return empty.
        return cKDTree(np.array([[1e9, 1e9, 1e9]])), np.array([-1], dtype=np.int32)
    arr = np.asarray(pts, dtype=np.float64)
    return cKDTree(arr), np.asarray(owner, dtype=np.int32)


def _is_skippable_sidechain(aa: str, atom_name: str, do_not_count_cb: bool) -> bool:
    """Should this atom be skipped when looking for sidechain clashes?

    Matches confind.cpp:622-624: backbone atoms are always skipped, and CB is
    skipped for non-ALA residues when ``do_not_count_cb`` is true (the C++
    default).
    """
    if atom_name in _BACKBONE_NAMES:
        return True
    if do_not_count_cb and aa != "ALA" and atom_name == "CB":
        return True
    return False


def build_position_rotamers(
    positions: list[Position],
    library: RotamerLibrary,
    *,
    aa_propensity: dict[str, float] | None = None,
    clash_dist: float = 2.0,
    do_not_count_cb: bool = True,
    aas: tuple[str, ...] = _DEFAULT_AAS,
    native_only: bool = False,
) -> list[PositionRotamers]:
    """Place and prune rotamers for every position.

    Parameters
    ----------
    positions
        Output of :func:`pyconfind.structure.positions_from_atoms`.
    library
        Loaded rotamer library.
    aa_propensity
        Override the default AA propensities (only used downstream by the
        contact-degree calculation; kept on the rotamer for convenience).
    clash_dist
        Backbone-clash distance cutoff in Å. C++ default 2.0.
    do_not_count_cb
        If ``True`` (C++ default), the CB atom of non-ALA residues is not
        counted for backbone clashes.
    aas
        Which AA identities to try at each position. The default matches the
        C++ binary's behavior of substituting in all 18 non-Gly/Pro AAs.
    native_only
        New mode (not in C++): only place rotamers of the native AA at each
        position, instead of substituting in all 18 AAs. The library still
        provides all rotamers for that AA; we just skip the substitutions.
    """
    propensity = aa_propensity or DEFAULT_AA_PROPENSITY
    tree, owner = build_clash_tree(positions)
    results: list[PositionRotamers] = []
    for pos in positions:
        rotamers_here, perm_contacts, frac_pruned, num_placed = _process_position(
            pos,
            library,
            tree,
            owner,
            aas=(pos.resname,) if native_only else aas,
            propensity=propensity,
            clash_dist=clash_dist,
            do_not_count_cb=do_not_count_cb,
        )
        results.append(
            PositionRotamers(
                position=pos,
                rotamers=tuple(rotamers_here),
                permanent_contacts=frozenset(perm_contacts),
                fraction_pruned=frac_pruned,
                num_rotamers_placed=num_placed,
            )
        )
    return results


def _process_position(
    pos: Position,
    library: RotamerLibrary,
    tree: cKDTree,
    owner: np.ndarray,
    *,
    aas: tuple[str, ...],
    propensity: dict[str, float],
    clash_dist: float,
    do_not_count_cb: bool,
) -> tuple[list[SurvivingRotamer], set[int], float, int]:
    surviving: list[SurvivingRotamer] = []
    perm: set[int] = set()
    total_placed = 0
    total_survived = 0
    for aa in aas:
        tmpl = library.residues.get(aa)
        if tmpl is None:
            continue
        if library.is_backbone_dependent:
            confs, weights = library.rotamers_for(aa, pos.phi, pos.psi)
        else:
            confs, weights = library.rotamers_for(aa)
        # Skip if the backbone is missing parents this template requires.
        if not _has_required_parents(tmpl, pos.backbone):
            continue
        placed = place_rotamers(tmpl, pos.backbone, confs=confs, weights=weights)
        total_placed += placed.coords.shape[0]
        # Vectorize the clash check across all rotamers of this identity.
        # For each rotamer, walk the relevant sidechain atoms; on the first
        # clash, mark for prune.
        sidechain_atom_idx = [
            i
            for i, name in enumerate(placed.atom_names)
            if not _is_skippable_sidechain(aa, name, do_not_count_cb)
        ]
        if not sidechain_atom_idx:
            # ALA where CB is skipped — i.e., GLY-like; no sidechain to check.
            sidechain_atom_idx = []
        # Per-rotamer clash detection + ALA permanent-contact accounting.
        pruned = np.zeros(placed.coords.shape[0], dtype=bool)
        for sc_idx in sidechain_atom_idx:
            coords = placed.coords[:, sc_idx, :]
            # Query each coordinate independently; cKDTree query_ball_point
            # accepts an array.
            neighbors = tree.query_ball_point(coords, r=clash_dist)
            for r, nbrs in enumerate(neighbors):
                if pruned[r]:
                    continue
                for n in nbrs:
                    if owner[n] == pos.index:
                        continue
                    pruned[r] = True
                    if aa == "ALA":
                        perm.add(int(owner[n]))
                    else:
                        break
        # Record surviving rotamers.
        sc_names = tuple(placed.atom_names[i] for i in sidechain_atom_idx)
        bb_names = tuple(
            n for n in placed.atom_names
            if n in _BACKBONE_NAMES
            or (do_not_count_cb and aa != "ALA" and n == "CB")
        )
        bb_idx = [placed.atom_names.index(n) for n in bb_names]
        # rot_id mirrors the C++ ``r`` loop index — i.e., index within the
        # rotamer list as passed to the placer.
        for r in range(placed.coords.shape[0]):
            if pruned[r]:
                continue
            sc_xyz = placed.coords[r, sidechain_atom_idx, :].copy()
            bb_xyz = placed.coords[r, bb_idx, :].copy()
            surviving.append(
                SurvivingRotamer(
                    aa=aa,
                    rot_id=r,
                    weight=float(placed.weights[r]),
                    aa_prop=propensity.get(aa, 0.0),
                    sidechain_atoms=sc_names,
                    sidechain_xyz=sc_xyz,
                    backbone_atoms=bb_names,
                    backbone_xyz=bb_xyz,
                )
            )
            total_survived += 1
    # When no rotamers can be placed (e.g. CA-only input where N/C are absent),
    # the C++ writes -nan via 0/0. Match that for downstream output parity.
    frac = float("nan") if total_placed == 0 else 1.0 - total_survived / total_placed
    return surviving, perm, frac, total_placed


def _has_required_parents(
    tmpl: ResidueICTemplate, backbone: dict[str, np.ndarray]
) -> bool:
    """Every IC parent atom name in ``tmpl`` must either be a placed atom or
    be present in ``backbone``."""
    placed = set(tmpl.placed)
    for row in tmpl.parents:
        for name in row:
            if name not in placed and name not in backbone:
                return False
    return True
