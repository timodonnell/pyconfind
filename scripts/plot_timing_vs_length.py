#!/usr/bin/env python3
"""Two-panel runtime-vs-sequence-length plot for the README.

Consumes ``docs/timing_results.json`` (a dict of ``{cpp_load_seconds, rows}``
where each row is ``{pdb, nres, py_numpy, py_numba, cpp,
py_numpy_native, py_numba_native}``). Library load time is excluded for both
implementations.

Left panel — ``native_only=False``:
    pyconfind (numpy), pyconfind (numba), ConFind (original)

Right panel — ``native_only=True`` (the C++ binary lacks this mode):
    pyconfind (numpy), pyconfind (numba)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PY_NUMPY = "#2563eb"  # blue
PY_NUMBA = "#16a34a"  # green
CPP      = "#dc2626"  # red


def _scatter(ax, x, y, color, label) -> None:
    order = np.argsort(x)
    x, y = x[order], y[order]
    ax.plot(x, y, "-", color=color, lw=1.0, alpha=0.6)
    ax.scatter(x, y, s=36, color=color, alpha=0.85, label=label, zorder=3,
               edgecolors="white", linewidths=0.6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("docs/timing_results.json"))
    ap.add_argument("--out",  type=Path, default=Path("docs/timing_vs_length.png"))
    args = ap.parse_args()

    payload = json.loads(args.data.read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    nres   = np.array([r["nres"] for r in rows], dtype=float)
    py_np  = np.array([r["py_numpy"] for r in rows], dtype=float)
    py_nb  = np.array([r["py_numba"] for r in rows], dtype=float)
    cpp    = np.array([r["cpp"] for r in rows], dtype=float)
    py_np_n = np.array([r["py_numpy_native"] for r in rows], dtype=float)
    py_nb_n = np.array([r["py_numba_native"] for r in rows], dtype=float)

    fig, (ax_full, ax_nat) = plt.subplots(1, 2, figsize=(13, 5.2))

    _scatter(ax_full, nres, py_np, PY_NUMPY, "pyconfind (numpy)")
    _scatter(ax_full, nres, py_nb, PY_NUMBA, "pyconfind (numba)")
    _scatter(ax_full, nres, cpp,   CPP,      "ConFind (original)")
    ax_full.set_title("native_only=False")

    _scatter(ax_nat, nres, py_np_n, PY_NUMPY, "pyconfind (numpy)")
    _scatter(ax_nat, nres, py_nb_n, PY_NUMBA, "pyconfind (numba)")
    ax_nat.set_title("native_only=True")

    for ax in (ax_full, ax_nat):
        ax.set_xlabel("Sequence length (residues)")
        ax.set_ylabel("Runtime per structure (s)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper left", frameon=True)

    fig.suptitle(
        "Per-structure runtime vs. sequence length (rotamer library pre-loaded)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
