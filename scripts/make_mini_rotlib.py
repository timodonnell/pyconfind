#!/usr/bin/env python3
"""Generate a small rotamer-library fixture for CI from the full library.

Keeps every amino acid's IC template (MOBI/DEFI) intact but truncates each to
the first ``--keep`` rotamers, and emits a BEBL with a single wildcard
``BIN * *`` per residue. The result is a valid, self-consistent library a few
tens of KB in size — enough to exercise the full place → prune → contact →
output pipeline in CI without committing the 100+ MB production library.

The numeric output of pyconfind on this fixture is *not* meant to match the
C++ reference (that needs the full library); it is snapshotted by
``tests/test_mini_pipeline.py`` purely as a regression guard.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def truncate_ebl(src: Path, dst: Path, keep: int) -> dict[str, int]:
    """Write a truncated EBL.out; return {resname: n_kept} for the BEBL."""
    out_lines: list[str] = []
    kept: dict[str, int] = {}
    cur: str | None = None
    n_conf = 0
    weights_line: list[str] | None = None
    pending_weights_resi: str | None = None

    def flush_weights() -> None:
        nonlocal weights_line, pending_weights_resi
        if weights_line is not None and pending_weights_resi is not None:
            n = kept[pending_weights_resi]
            out_lines.append("WEIGHTS " + " ".join(weights_line[1 : 1 + n]))
            weights_line = None
            pending_weights_resi = None

    for raw in src.read_text().splitlines():
        s = raw.strip()
        if s.startswith("LIBRARY"):
            out_lines.append(raw)
            continue
        if s.startswith("RESI"):
            flush_weights()
            cur = s.split()[1]
            n_conf = 0
            kept[cur] = 0
            out_lines.append(raw)
            continue
        if cur is None:
            continue
        if s.startswith("MOBI") or s.startswith("DEFI"):
            out_lines.append(raw)
        elif s.startswith("CONF"):
            if n_conf < keep:
                out_lines.append(raw)
                n_conf += 1
                kept[cur] = n_conf
        elif s.startswith("WEIGHTS"):
            weights_line = s.split()
            pending_weights_resi = cur
    flush_weights()
    dst.write_text("\n".join(out_lines) + "\n")
    return kept


def write_bebl(dst: Path, kept: dict[str, int], phibin: int = 10, psibin: int = 10) -> None:
    lines = [f"PHIBIN {phibin}", f"PSIBIN {psibin}", ""]
    for resi, n in kept.items():
        if n == 0:
            continue
        lines.append(f"RESI {resi}")
        lines.append("BIN * *")
        lines.append(f"LEVNUM {n}")
        lines.append("CONFIDX " + " ".join(str(i) for i in range(n)))
        lines.append("")
    dst.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Full rotlib dir.")
    parser.add_argument("--dst", type=Path, required=True, help="Output mini rotlib dir.")
    parser.add_argument("--keep", type=int, default=5, help="Rotamers per AA.")
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    kept = truncate_ebl(args.src / "EBL.out", args.dst / "EBL.out", args.keep)
    write_bebl(args.dst / "BEBL.out", kept)
    print(f"Wrote mini library to {args.dst} ({sum(kept.values())} rotamers across {len(kept)} AAs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
