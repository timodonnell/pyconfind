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


def _fit_offset_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit ``y = c + a * x^b`` (fixed overhead + algorithmic term).

    Returns ``(c, a, b)``. Falls back to a pure power law (c=0) if the
    nonlinear fit fails to converge.
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        a, b = _fit_power_law(x, y)
        return 0.0, a, b

    def model(n: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
        return c + a * np.power(n, b)

    a0, b0 = _fit_power_law(x, y)
    try:
        popt, _ = curve_fit(
            model, x, y, p0=[0.0, a0, b0],
            bounds=([0, 0, 0.3], [np.inf, np.inf, 3.0]), maxfev=20000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return 0.0, a0, b0


def _fit_label(c: float, a: float, b: float) -> str:
    """Format a fit as ``a·N^b`` (plus a ``c +`` prefix only if non-negligible)."""
    term = f"{a:.1e}·N^{b:.2f}"
    return f"{c:.1f}s + {term}" if c >= 0.05 else f"∝ N^{b:.2f}"


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

    # Offset power-law fit (fixed overhead + algorithmic term) separates each
    # implementation's constant startup cost from its true scaling exponent.
    py_c, py_a, py_b = _fit_offset_power_law(nres, py)
    cpp_c, cpp_a, cpp_b = _fit_offset_power_law(nres, cpp)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5.5))
    xx = np.linspace(nres.min(), nres.max(), 200)

    for ax in (ax_lin, ax_log):
        ax.scatter(nres, cpp, s=28, color="#dc2626", alpha=0.8,
                   label="C++ confind (incl. per-run library load)", zorder=3)
        ax.scatter(nres, py, s=28, color="#2563eb", alpha=0.8,
                   label="pyconfind (library amortized)", zorder=3)
        ax.plot(xx, cpp_c + cpp_a * xx**cpp_b, "--", color="#dc2626", lw=1.2,
                label=f"C++ fit: {_fit_label(cpp_c, cpp_a, cpp_b)}")
        ax.plot(xx, py_c + py_a * xx**py_b, "--", color="#2563eb", lw=1.2,
                label=f"pyconfind fit: {_fit_label(py_c, py_a, py_b)}")
        ax.set_xlabel("Sequence length (residues)")
        ax.set_ylabel("Runtime per structure (s)")
        ax.grid(True, alpha=0.25)

    ax_lin.set_title("Linear scale")
    ax_lin.legend(fontsize=8)
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_title("Log-log scale")
    ax_log.legend(fontsize=8)

    fig.suptitle(
        "confind runtime vs. sequence length\n"
        "fit: fixed overhead + a·N^b (separates per-run startup from scaling); "
        "pyconfind library load is amortized, C++ reloads it each run",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"\npyconfind: runtime ≈ {py_c:.2f}s + {py_a:.2e} · N^{py_b:.2f}")
    print(f"C++:       runtime ≈ {cpp_c:.2f}s + {cpp_a:.2e} · N^{cpp_b:.2f}")
    # Fair algorithmic comparison: ratio of the N-dependent coefficients at the
    # largest N tested (overhead-subtracted).
    n_max = nres.max()
    py_algo = py_a * n_max**py_b
    cpp_algo = cpp_a * n_max**cpp_b
    print(f"\nAt N={int(n_max)}: algorithmic time (overhead-subtracted) "
          f"py={py_algo:.1f}s vs cpp={cpp_algo:.1f}s → {cpp_algo/py_algo:.1f}×")
    print(f"Median wall-clock speedup (cpp/py, as invoked): {np.median(cpp / py):.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
