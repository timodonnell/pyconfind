"""Rotamer library parser for MSL-format Dunbrack-style libraries.

A library consists of two files:

* ``EBL.out`` — the rotamer pool. For each amino acid:

  * ``RESI <AA>``
  * ``MOBI <atoms...>``                       — atoms placed by the IC table
  * ``DEFI <four atoms>`` × M                 — dihedrals (one per mobile atom)
  * ``DEFI <three atoms>`` × M                — bond angles
  * ``DEFI <two atoms>`` × M                  — bond lengths
  * ``CONF v1 v2 ... v(3M)`` × R              — one row per rotamer
  * ``WEIGHTS w1 ... wR``

  For the bb-indep case, the weights are global; for the bb-dep case, each
  conformation belongs to a single (phi, psi) bin and the weight is
  conditional on that bin.

* ``BEBL.out`` — backbone-dependent indexing. Header lines ``PHIBIN N`` and
  ``PSIBIN N`` give bin widths in degrees. Each amino acid then has a block of

  * ``BIN <phi_lo> <psi_lo>``
  * ``LEVNUM <n>``
  * ``CONFIDX i1 i2 ... in``                  — indices into the EBL pool

  where the conf indices select the rotamers active for that bin.

If only an ``EBL.out`` file is given, the library is treated as
backbone-independent (matches the C++ ``--rLib <file>`` mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidueICTemplate:
    """Internal-coord template for one residue type.

    ``placed`` is the ordered list of atoms placed by the IC table, in the
    order they are built. The MSL ``MOBI`` line is a *set* of mobile atom names;
    the actual placement order is determined by the ordering of the DEFI lines.

    ``parents`` is an array of shape ``(M, 3)`` of atom-name strings: for
    placed atom ``k``, ``parents[k]`` are (a, b, c) such that the placement
    uses dihedral a-b-c-placed[k], angle b-c-placed[k], and bond length
    c-placed[k].

    ``confs`` has shape ``(R, M, 3)`` with the (dihedral°, angle°, bond Å)
    values for each of the ``R`` rotamers. ``weights`` has length ``R``.
    """

    resname: str
    placed: tuple[str, ...]
    parents: np.ndarray  # (M, 3) of <U4 atom names
    confs: np.ndarray    # (R, M, 3) float64: (dihedral, angle, bond)
    weights: np.ndarray  # (R,) float64


@dataclass(frozen=True, slots=True)
class RotamerLibrary:
    """A complete rotamer library, possibly backbone-dependent."""

    name: str
    residues: dict[str, ResidueICTemplate]
    # bb-dep fields; absent for bb-indep libraries
    phi_bin: float | None = None
    psi_bin: float | None = None
    # mapping (resname, phi_bin_idx, psi_bin_idx) -> array of CONFIDX
    bin_index: dict[tuple[str, int, int], np.ndarray] | None = None

    @property
    def is_backbone_dependent(self) -> bool:
        return self.bin_index is not None

    def rotamers_for(
        self, resname: str, phi: float | None = None, psi: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(confs, weights)`` for the given residue (and phi/psi if bb-dep).

        For backbone-dependent libraries, passing ``phi=None`` or ``psi=None``
        selects the wildcard ``BIN * *`` fallback that the library defines for
        terminal residues where the dihedral is undefined.
        """
        tmpl = self.residues[resname]
        if not self.is_backbone_dependent:
            return tmpl.confs, tmpl.weights
        assert self.phi_bin is not None and self.psi_bin is not None
        assert self.bin_index is not None
        if phi is None or psi is None:
            pi, si = -1, -1
        else:
            pi = _bin_index(phi, self.phi_bin)
            si = _bin_index(psi, self.psi_bin)
        idx = self.bin_index[(resname, pi, si)]
        return tmpl.confs[idx], tmpl.weights[idx]


def _bin_index(angle: float, width: float) -> int:
    """Bin a dihedral angle in degrees onto a lattice with given width.

    Bins start at -180 with the given width; -180 maps to 0, -180 + width to 1,
    etc. Out-of-range angles wrap into [-180, 180).
    """
    # Wrap to [-180, 180)
    a = ((angle + 180.0) % 360.0) - 180.0
    return int((a - (-180.0)) // width)


def parse_ebl(path: str | Path) -> dict[str, ResidueICTemplate]:
    """Parse an ``EBL.out``-style file into IC templates keyed by residue name."""
    text = Path(path).read_text()
    out: dict[str, ResidueICTemplate] = {}
    # Split into residue blocks
    current: dict | None = None

    def finalize() -> None:
        if current is None:
            return
        defis = current["defis"]
        # Group DEFIs by arity (4 = dihedral, 3 = angle, 2 = bond length).
        # Their order within each arity group defines the placement order: the
        # k-th dihedral, k-th angle, and k-th bond all describe the same mobile
        # atom (the one named in the last position of each). MOBI is a *set* of
        # mobile atom names, not necessarily in placement order (PRO is an
        # example).
        dih = [d for d in defis if len(d) == 4]
        ang = [d for d in defis if len(d) == 3]
        bnd = [d for d in defis if len(d) == 2]
        n = len(dih)
        if not (len(ang) == n and len(bnd) == n):
            raise ValueError(
                f"RESI {current['name']}: DEFI arity counts mismatch "
                f"(dih={len(dih)}, ang={len(ang)}, bnd={len(bnd)})"
            )
        placed = tuple(d[-1] for d in dih)
        for k in range(n):
            if ang[k][-1] != placed[k]:
                raise ValueError(
                    f"RESI {current['name']}: angle DEFI #{k} ends in {ang[k][-1]}, "
                    f"expected {placed[k]}"
                )
            if bnd[k][-1] != placed[k]:
                raise ValueError(
                    f"RESI {current['name']}: bond DEFI #{k} ends in {bnd[k][-1]}, "
                    f"expected {placed[k]}"
                )
        mobile_set = set(current["mobile"])
        if set(placed) != mobile_set:
            raise ValueError(
                f"RESI {current['name']}: MOBI {sorted(mobile_set)} != "
                f"placed atoms {sorted(set(placed))}"
            )

        parents = np.array(
            [[d[0], d[1], d[2]] for d in dih], dtype="<U4"
        )

        confs_raw = current["confs"]
        if not confs_raw:
            raise ValueError(f"RESI {current['name']}: no CONF rows")
        confs_flat = np.asarray(confs_raw, dtype=np.float64)
        if confs_flat.shape[1] != 3 * n:
            raise ValueError(
                f"RESI {current['name']}: CONF row width {confs_flat.shape[1]} != {3*n}"
            )
        # CONF layout: first n dihedrals, next n angles, last n bond lengths
        confs = np.empty((confs_flat.shape[0], n, 3), dtype=np.float64)
        confs[:, :, 0] = confs_flat[:, :n]
        confs[:, :, 1] = confs_flat[:, n:2 * n]
        confs[:, :, 2] = confs_flat[:, 2 * n:]

        weights = np.asarray(current["weights"], dtype=np.float64)
        if weights.size != confs.shape[0]:
            raise ValueError(
                f"RESI {current['name']}: {weights.size} weights for {confs.shape[0]} CONFs"
            )

        out[current["name"]] = ResidueICTemplate(
            resname=current["name"],
            placed=placed,
            parents=parents,
            confs=confs,
            weights=weights,
        )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("LIBRARY"):
            continue
        if line.startswith("RESI"):
            finalize()
            parts = line.split()
            current = {
                "name": parts[1],
                "mobile": [],
                "defis": [],
                "confs": [],
                "weights": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("MOBI"):
            current["mobile"] = line.split()[1:]
        elif line.startswith("DEFI"):
            current["defis"].append(tuple(line.split()[1:]))
        elif line.startswith("CONF"):
            current["confs"].append([float(x) for x in line.split()[1:]])
        elif line.startswith("WEIGHTS"):
            current["weights"] = [float(x) for x in line.split()[1:]]
    finalize()
    return out


def parse_bebl(
    path: str | Path,
) -> tuple[float, float, dict[tuple[str, int, int], np.ndarray]]:
    """Parse a ``BEBL.out`` file.

    Returns ``(phi_bin_width, psi_bin_width, bin_index)`` where ``bin_index``
    maps ``(resname, phi_bin_index, psi_bin_index)`` to a 1-D NumPy array of
    CONFIDX values selecting rotamers from the EBL pool.
    """
    text = Path(path).read_text()
    phi_bin: float | None = None
    psi_bin: float | None = None
    index: dict[tuple[str, int, int], np.ndarray] = {}
    cur_resi: str | None = None
    cur_bin: tuple[int, int] | None = None
    cur_levnum: int | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("PHIBIN"):
            phi_bin = float(line.split()[1])
            continue
        if line.startswith("PSIBIN"):
            psi_bin = float(line.split()[1])
            continue
        if line.startswith("RESI"):
            cur_resi = line.split()[1]
            continue
        if line.startswith("BIN"):
            assert phi_bin is not None and psi_bin is not None
            parts = line.split()
            # ``BIN * *`` is the fallback used when phi or psi is undefined
            # (terminal residues etc.). We store it under bin index (-1, -1).
            if parts[1] == "*" or parts[2] == "*":
                cur_bin = (-1, -1)
            else:
                phi_lo = float(parts[1])
                psi_lo = float(parts[2])
                pi = _bin_index(phi_lo, phi_bin)
                si = _bin_index(psi_lo, psi_bin)
                cur_bin = (pi, si)
            continue
        if line.startswith("LEVNUM"):
            cur_levnum = int(line.split()[1])
            continue
        if line.startswith("CONFIDX"):
            assert cur_resi is not None and cur_bin is not None
            idx = np.fromiter(
                (int(x) for x in line.split()[1:]),
                dtype=np.int32,
            )
            if cur_levnum is not None and idx.size != cur_levnum:
                raise ValueError(
                    f"RESI {cur_resi} BIN {cur_bin}: CONFIDX count {idx.size} "
                    f"!= LEVNUM {cur_levnum}"
                )
            index[(cur_resi, cur_bin[0], cur_bin[1])] = idx
            cur_levnum = None
            continue

    if phi_bin is None or psi_bin is None:
        raise ValueError(f"{path}: missing PHIBIN/PSIBIN header")
    return phi_bin, psi_bin, index


def load_library(path: str | Path) -> RotamerLibrary:
    """Load a rotamer library from a file or directory.

    * If ``path`` is a directory containing ``EBL.out`` and ``BEBL.out``, a
      backbone-dependent library is returned.
    * If ``path`` is a single ``EBL.out``-format file, a backbone-independent
      library is returned (using the global weights in that file).

    This mirrors the C++ ``--rLib`` argument handling.
    """
    p = Path(path)
    if p.is_dir():
        residues = parse_ebl(p / "EBL.out")
        phi_bin, psi_bin, bin_index = parse_bebl(p / "BEBL.out")
        return RotamerLibrary(
            name=p.name,
            residues=residues,
            phi_bin=phi_bin,
            psi_bin=psi_bin,
            bin_index=bin_index,
        )
    residues = parse_ebl(p)
    return RotamerLibrary(name=p.stem, residues=residues)
