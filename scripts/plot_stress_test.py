#!/usr/bin/env python3
"""Plot timing comparison from a stress-test run.

Reads either:

* ``--log <path>`` to ``stress_test.py``'s live tee'd log (works mid-run)
* ``--summary <path>`` to the final ``summary.json``

and produces a scatter of Python vs. C++ runtime per structure, colored by
source (PDB vs. AFDB), plus a histogram of speedup ratios.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Row:
    label: str
    pdb: str
    total_lines: int
    diff_lines: int
    py_seconds: float
    cpp_seconds: float
    status: str = "match"


_LOG_LINE = re.compile(
    r"\[\s*(\d+)/\d+\]\s+(\S)\s+(\S+)\s+\(\s*(\d+)L,\s*(\d+)\s+diff,\s+"
    r"py\s+([\d.]+)s,\s+cpp\s+([\d.]+)s,\s+total\s+([\d.]+)s\)"
)
_SECTION = re.compile(r"^Running (\S+) \(")
_TAG_TO_STATUS = {"✓": "match", "✗": "diff", "C": "error_cpp", "P": "error_py"}


def parse_log(path: Path) -> list[Row]:
    rows: list[Row] = []
    label = "PDB"
    for raw in path.read_text().splitlines():
        m = _SECTION.search(raw)
        if m:
            label = m.group(1)
            continue
        m = _LOG_LINE.search(raw)
        if not m:
            continue
        _, tag, name, total, diff, py_s, cpp_s, _ = m.groups()
        rows.append(
            Row(
                label=label,
                pdb=name,
                total_lines=int(total),
                diff_lines=int(diff),
                py_seconds=float(py_s),
                cpp_seconds=float(cpp_s),
                status=_TAG_TO_STATUS.get(tag, "?"),
            )
        )
    return rows


def parse_summary(path: Path) -> list[Row]:
    payload = json.loads(path.read_text())
    return [
        Row(
            label=r["label"], pdb=r["pdb"],
            total_lines=int(r.get("total_lines") or 0),
            diff_lines=int(r.get("diff_lines") or 0),
            py_seconds=float(r["py_seconds"]),
            cpp_seconds=float(r["cpp_seconds"]),
            status=r["status"],
        )
        for r in payload["results"]
    ]


def make_plot(rows: list[Row], out_path: Path, title_suffix: str = "") -> None:
    timed = [r for r in rows if r.py_seconds > 0 and r.cpp_seconds > 0]
    if not timed:
        print("No timing rows yet — nothing to plot.", file=sys.stderr)
        return

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.5))

    colors = {"PDB": "#2563eb", "AFDB": "#dc2626"}
    markers = {"match": "o", "diff": "x", "error_cpp": "s", "error_py": "^"}

    for label in ("PDB", "AFDB"):
        for status in ("match", "diff", "error_cpp", "error_py"):
            pts = [r for r in timed if r.label == label and r.status == status]
            if not pts:
                continue
            ax_scatter.scatter(
                [r.cpp_seconds for r in pts],
                [r.py_seconds for r in pts],
                c=colors.get(label, "gray"),
                marker=markers.get(status, "."),
                s=22, alpha=0.7,
                label=f"{label} {status}" if status != "match" else label,
            )

    # Equal-time reference + 2× and 0.5× speedup guides.
    lo = min(min(r.cpp_seconds for r in timed), min(r.py_seconds for r in timed)) * 0.5
    hi = max(max(r.cpp_seconds for r in timed), max(r.py_seconds for r in timed)) * 2.0
    ax_scatter.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="equal (1×)")
    ax_scatter.plot([lo, hi], [2 * lo, 2 * hi], ":", color="lightgray", lw=1)
    ax_scatter.plot([lo, hi], [0.5 * lo, 0.5 * hi], ":", color="lightgray", lw=1)
    ax_scatter.text(hi * 0.9, hi * 1.8, "py 2×", fontsize=8, color="gray", ha="right")
    ax_scatter.text(hi * 0.9, hi * 0.45, "py 0.5×", fontsize=8, color="gray", ha="right")

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_xlabel("C++ confind runtime (s, log)")
    ax_scatter.set_ylabel("pyconfind runtime (s, log)")
    ax_scatter.set_title(f"Per-structure runtime{title_suffix}")
    ax_scatter.legend(loc="upper left", fontsize=9)
    ax_scatter.grid(True, which="both", alpha=0.2)

    # Histogram of speedup ratios (C++/Python: higher = pyconfind faster).
    ratios = {"PDB": [], "AFDB": []}
    for r in timed:
        if r.py_seconds > 0:
            ratios.setdefault(r.label, []).append(r.cpp_seconds / r.py_seconds)
    bins = 24
    ax_hist.hist(
        [ratios.get("PDB", []), ratios.get("AFDB", [])],
        bins=bins, label=["PDB", "AFDB"],
        color=[colors["PDB"], colors["AFDB"]], alpha=0.7,
    )
    ax_hist.axvline(1.0, color="gray", linestyle="--", lw=1, label="parity")
    ax_hist.set_xlabel("Speedup (C++ time ÷ Python time)")
    ax_hist.set_ylabel("# structures")
    ax_hist.set_title("Per-structure speedup distribution")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        f"pyconfind vs. C++ confind — {len(timed)} structures",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Print a numeric summary.
    py_total = sum(r.py_seconds for r in timed)
    cpp_total = sum(r.cpp_seconds for r in timed)
    print(f"\n{len(timed)} structures timed")
    print(f"  Total py:  {py_total:7.1f}s")
    print(f"  Total cpp: {cpp_total:7.1f}s")
    print(f"  Overall speedup: {cpp_total / py_total:.2f}× (cpp / py)")
    for label in ("PDB", "AFDB"):
        subset = [r for r in timed if r.label == label]
        if not subset:
            continue
        py_t = sum(r.py_seconds for r in subset)
        cpp_t = sum(r.cpp_seconds for r in subset)
        match = sum(1 for r in subset if r.status == "match")
        print(f"  {label}: {len(subset)} structures, {match} byte-identical, "
              f"speedup {cpp_t / py_t:.2f}×")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", type=Path, help="Path to stress_test.py's live log.")
    src.add_argument("--summary", type=Path, help="Path to summary.json.")
    parser.add_argument("--out", type=Path, default=Path("stress_test_plot.png"))
    parser.add_argument("--title-suffix", type=str, default="")
    args = parser.parse_args()

    rows = parse_summary(args.summary) if args.summary else parse_log(args.log)
    make_plot(rows, args.out, args.title_suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
