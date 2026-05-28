#!/usr/bin/env python3
"""Plot runtime as a function of sequence length (residue count).

Consumes the JSON produced by the timing harness — a list of
``{"pdb", "nres", "py", "cpp"}`` records — and plots pyconfind vs. C++
runtime against residue count, with power-law fits to show the scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``y = a * x^b``; return ``(a, b)``."""
    mask = (x > 0) & (y > 0)
    b, log_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(np.exp(log_a)), float(b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/timing_results.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/timing_vs_length.png"))
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())
    nres = np.array([r["nres"] for r in rows], dtype=float)
    py = np.array([r["py"] for r in rows], dtype=float)
    cpp = np.array([r["cpp"] for r in rows], dtype=float)

    order = np.argsort(nres)
    nres, py, cpp = nres[order], py[order], cpp[order]

    py_a, py_b = _fit_power_law(nres, py)
    cpp_a, cpp_b = _fit_power_law(nres, cpp)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5.5))
    xx = np.linspace(nres.min(), nres.max(), 200)

    for ax in (ax_lin, ax_log):
        ax.scatter(nres, cpp, s=28, color="#dc2626", alpha=0.8,
                   label="C++ confind (incl. per-run library load)", zorder=3)
        ax.scatter(nres, py, s=28, color="#2563eb", alpha=0.8,
                   label="pyconfind (library amortized)", zorder=3)
        ax.plot(xx, cpp_a * xx**cpp_b, "--", color="#dc2626", lw=1.2,
                label=f"C++ fit ∝ N^{cpp_b:.2f}")
        ax.plot(xx, py_a * xx**py_b, "--", color="#2563eb", lw=1.2,
                label=f"pyconfind fit ∝ N^{py_b:.2f}")
        ax.set_xlabel("Sequence length (residues)")
        ax.set_ylabel("Runtime per structure (s)")
        ax.grid(True, alpha=0.25)

    ax_lin.set_title("Linear scale")
    ax_lin.legend(fontsize=8.5)
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_title("Log-log scale (slope = scaling exponent)")
    ax_log.legend(fontsize=8.5)

    fig.suptitle(
        "confind runtime vs. sequence length\n"
        "pyconfind times exclude the one-time ~3.4 s library load (amortized "
        "in batch use); the C++ binary reloads it every invocation",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"\npyconfind: runtime ≈ {py_a:.2e} · N^{py_b:.2f}")
    print(f"C++:       runtime ≈ {cpp_a:.2e} · N^{cpp_b:.2f}")
    print(f"\nMedian speedup (cpp/py): {np.median(cpp / py):.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
