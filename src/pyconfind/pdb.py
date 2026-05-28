"""Minimal PDB parser for pyconfind.

The PDB ``ATOM`` / ``HETATM`` record is a fixed-column format. We extract just
what confind needs (chain, resnum, icode, resname, atomname, altloc, xyz) into
NumPy arrays. Multi-model files (e.g. NMR) yield only the first model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Residue names treated as "protein" by the C++ confind (see confind.cpp:840).
# Includes the standard 20 plus HIS-variant names (HSD/HSE/HSC/HSP), MSE, and a
# handful of phospho/modified residues.
LEGAL_RESIDUE_NAMES: frozenset[str] = frozenset({
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
    "HSD", "HSE", "HSC", "HSP", "MSE", "CSO", "HIP", "SEC", "SEP", "TPO",
    "PTR",
})


@dataclass(frozen=True)
class Atoms:
    """Per-atom NumPy arrays. All arrays have the same length ``N``.

    ``position_index`` groups atoms into positions: atoms with the same value
    belong to the same (chain, resnum, icode) position. ``identity_index``
    groups atoms within a position by resname (a Position may have multiple
    Identities if a PDB has overlapping alternative residue types — confind's
    ``example0000.pdb`` puts both ILE and LEU at position 2).

    Positions appear in their input PDB order; identities within a position
    appear in input order; atoms within an identity appear in input order.
    """

    chain: np.ndarray       # (N,) <U2
    resnum: np.ndarray      # (N,) int32
    icode: np.ndarray       # (N,) <U1
    resname: np.ndarray     # (N,) <U4
    name: np.ndarray        # (N,) <U4 — stripped atom names
    altloc: np.ndarray      # (N,) <U1
    element: np.ndarray     # (N,) <U2 — stripped element symbol
    xyz: np.ndarray         # (N, 3) float64
    occupancy: np.ndarray   # (N,) float32
    position_index: np.ndarray   # (N,) int32 — 0..P-1
    identity_index: np.ndarray   # (N,) int32 — 0..(per-position-K - 1)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def num_atoms(self) -> int:
        return len(self)

    @property
    def num_positions(self) -> int:
        return int(self.position_index.max()) + 1 if len(self) else 0


def read_pdb(
    path: str | Path,
    *,
    legal_only: bool = True,
    altloc: str = "A",
    renumber: bool = False,
) -> Atoms:
    """Read a PDB file and return atom arrays.

    Parameters
    ----------
    path
        PDB file path.
    legal_only
        If ``True`` (the C++ default), drop atoms whose residue name is not in
        :data:`LEGAL_RESIDUE_NAMES`.
    altloc
        Which alternate-location indicator to keep. Atoms with altloc equal to
        either ``" "`` or this value are kept; others are dropped. The default
        ``"A"`` matches conventional behavior.
    renumber
        If ``True``, renumber residues within each chain starting from 1 (the
        ``--ren`` flag in C++ confind).
    """
    chains: list[str] = []
    resnums: list[int] = []
    icodes: list[str] = []
    resnames: list[str] = []
    names: list[str] = []
    altlocs: list[str] = []
    elements: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    occs: list[float] = []

    in_model = True  # accept atoms until first ENDMDL
    saw_endmdl = False
    with Path(path).open() as fh:
        for raw in fh:
            rec = raw[:6]
            if rec == "ENDMDL":
                in_model = False
                saw_endmdl = True
                continue
            if rec == "MODEL ":
                # If we've already seen an ENDMDL, ignore subsequent models.
                if saw_endmdl:
                    continue
                in_model = True
                continue
            if not in_model:
                continue
            if rec != "ATOM  " and rec != "HETATM":
                continue
            # Pad with spaces in case the line is short.
            line = raw.rstrip("\n").ljust(80)
            altloc_c = line[16]
            if altloc_c not in (" ", altloc):
                continue
            resname = line[17:20].strip()
            if legal_only and resname not in LEGAL_RESIDUE_NAMES:
                continue
            chain = line[21].strip() or "_"
            resnum = int(line[22:26])
            icode = line[26].strip()
            atom_name = line[12:16].strip()
            element = line[76:78].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                occ = float(line[54:60])
            except ValueError:
                occ = 1.0
            chains.append(chain)
            resnums.append(resnum)
            icodes.append(icode)
            resnames.append(resname)
            names.append(atom_name)
            altlocs.append(altloc_c.strip())
            elements.append(element)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            occs.append(occ)

    n = len(xs)
    chain_arr = np.asarray(chains, dtype="<U2") if n else np.zeros(0, dtype="<U2")
    resnum_arr = np.asarray(resnums, dtype=np.int32) if n else np.zeros(0, dtype=np.int32)
    icode_arr = np.asarray(icodes, dtype="<U1") if n else np.zeros(0, dtype="<U1")
    resname_arr = np.asarray(resnames, dtype="<U4") if n else np.zeros(0, dtype="<U4")
    name_arr = np.asarray(names, dtype="<U4") if n else np.zeros(0, dtype="<U4")
    altloc_arr = np.asarray(altlocs, dtype="<U1") if n else np.zeros(0, dtype="<U1")
    element_arr = np.asarray(elements, dtype="<U2") if n else np.zeros(0, dtype="<U2")
    xyz = np.column_stack([xs, ys, zs]).astype(np.float64) if n else np.zeros((0, 3))
    occ_arr = np.asarray(occs, dtype=np.float32) if n else np.zeros(0, dtype=np.float32)

    position_index, identity_index = _build_position_indices(
        chain_arr, resnum_arr, icode_arr, resname_arr
    )

    atoms = Atoms(
        chain=chain_arr,
        resnum=resnum_arr,
        icode=icode_arr,
        resname=resname_arr,
        name=name_arr,
        altloc=altloc_arr,
        element=element_arr,
        xyz=xyz,
        occupancy=occ_arr,
        position_index=position_index,
        identity_index=identity_index,
    )

    # Reorder positions to match MSL's Chain ordering (see _reorder_msl), then
    # renumber if requested — the C++ orders positions before renumbering.
    atoms = _reorder_msl(atoms)
    if renumber:
        atoms = _renumber_chains(atoms)
    return atoms


def _renumber_chains(atoms: Atoms) -> Atoms:
    """Renumber each chain's positions from 1 (the ``--ren`` flag).

    Operates on positions in their current order; atoms inherit their
    position's new number and lose insertion codes.
    """
    n = len(atoms)
    if n == 0:
        return atoms
    slices = position_iter(atoms)
    new_resnum = np.empty(n, dtype=np.int32)
    per_chain: dict[str, int] = {}
    for s in slices:
        ch = str(atoms.chain[s.start])
        per_chain[ch] = per_chain.get(ch, 0) + 1
        new_resnum[s] = per_chain[ch]
    pos_idx, id_idx = _build_position_indices(
        atoms.chain, new_resnum, np.zeros(n, dtype="<U1"), atoms.resname
    )
    return Atoms(
        chain=atoms.chain,
        resnum=new_resnum,
        icode=np.zeros(n, dtype="<U1"),
        resname=atoms.resname,
        name=atoms.name,
        altloc=atoms.altloc,
        element=atoms.element,
        xyz=atoms.xyz,
        occupancy=atoms.occupancy,
        position_index=pos_idx,
        identity_index=id_idx,
    )


def _reorder_msl(atoms: Atoms) -> Atoms:
    """Reorder atoms so positions follow MSL's Chain ordering.

    MSL inserts positions with a comparator (``sortByResnumIcodeAscending``)
    that — due to a bug comparing an insertion code against itself — reduces
    to: order by residue number ascending, and within a residue number place
    the blank-insertion-code residue first, with lettered insertion codes
    keeping their file order. Chains stay in first-appearance order. For the
    common case (ascending residue numbers, no insertion codes) this is a
    no-op; it only matters for protease/antibody numbering where the file
    lists insertion-coded residues out of ascending order.
    """
    n = len(atoms)
    if n == 0:
        return atoms
    slices = position_iter(atoms)
    # Chain first-appearance order.
    chain_rank: dict[str, int] = {}
    for s in slices:
        ch = str(atoms.chain[s.start])
        if ch not in chain_rank:
            chain_rank[ch] = len(chain_rank)
    # Stable sort key per position; the slice index (file order) breaks ties.
    order = sorted(
        range(len(slices)),
        key=lambda p: (
            chain_rank[str(atoms.chain[slices[p].start])],
            int(atoms.resnum[slices[p].start]),
            0 if str(atoms.icode[slices[p].start]) == "" else 1,
            p,
        ),
    )
    if order == list(range(len(slices))):
        return atoms  # already in MSL order — avoid copying
    new_index = np.concatenate([np.arange(slices[p].start, slices[p].stop) for p in order])
    pos_idx, id_idx = _build_position_indices(
        atoms.chain[new_index], atoms.resnum[new_index],
        atoms.icode[new_index], atoms.resname[new_index],
    )
    return Atoms(
        chain=atoms.chain[new_index],
        resnum=atoms.resnum[new_index],
        icode=atoms.icode[new_index],
        resname=atoms.resname[new_index],
        name=atoms.name[new_index],
        altloc=atoms.altloc[new_index],
        element=atoms.element[new_index],
        xyz=atoms.xyz[new_index],
        occupancy=atoms.occupancy[new_index],
        position_index=pos_idx,
        identity_index=id_idx,
    )


def _build_position_indices(
    chain: np.ndarray,
    resnum: np.ndarray,
    icode: np.ndarray,
    resname: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position_index, identity_index)`` arrays.

    Two atoms share a position if they have the same (chain, resnum, icode).
    Within a position, atoms share an identity if they also share a resname.
    Positions are numbered in input order; identities within a position are
    numbered in the order they appear.
    """
    n = chain.size
    pos_idx = np.empty(n, dtype=np.int32)
    id_idx = np.empty(n, dtype=np.int32)
    seen_pos: dict[tuple[str, int, str], int] = {}
    seen_id: dict[tuple[str, int, str, str], int] = {}
    pos_id_counter: dict[int, int] = {}
    next_pos = 0
    for i in range(n):
        key_p = (str(chain[i]), int(resnum[i]), str(icode[i]))
        if key_p in seen_pos:
            pi = seen_pos[key_p]
        else:
            pi = next_pos
            seen_pos[key_p] = pi
            pos_id_counter[pi] = 0
            next_pos += 1
        pos_idx[i] = pi
        key_id = (key_p[0], key_p[1], key_p[2], str(resname[i]))
        if key_id in seen_id:
            id_idx[i] = seen_id[key_id]
        else:
            id_idx[i] = pos_id_counter[pi]
            pos_id_counter[pi] += 1
            seen_id[key_id] = id_idx[i]
    return pos_idx, id_idx


def filter_atoms_by_position(atoms: Atoms, position_mask: np.ndarray) -> Atoms:
    """Return a new :class:`Atoms` keeping only atoms in selected positions.

    ``position_mask`` is a boolean array of length ``num_positions``. Position
    and identity indices are recomputed so they stay contiguous from 0.
    """
    keep = position_mask[atoms.position_index]
    pos_idx, id_idx = _build_position_indices(
        atoms.chain[keep], atoms.resnum[keep], atoms.icode[keep], atoms.resname[keep]
    )
    return Atoms(
        chain=atoms.chain[keep],
        resnum=atoms.resnum[keep],
        icode=atoms.icode[keep],
        resname=atoms.resname[keep],
        name=atoms.name[keep],
        altloc=atoms.altloc[keep],
        element=atoms.element[keep],
        xyz=atoms.xyz[keep],
        occupancy=atoms.occupancy[keep],
        position_index=pos_idx,
        identity_index=id_idx,
    )


def position_iter(atoms: Atoms) -> list[slice]:
    """Return one ``slice`` per position covering its atoms.

    Atoms within a position are contiguous because positions are assigned in
    input order. Useful for per-residue loops without re-grouping.
    """
    if len(atoms) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(atoms.position_index) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(atoms)]))
    return [slice(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
