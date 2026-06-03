#!/usr/bin/env python3
"""Run timings for the README runtime-vs-length plot.

Excludes the rotamer-library load from every measurement:

* pyconfind: the library is loaded once at the start of this process; only
  per-call analyze() time is recorded.
* C++ confind: the binary re-parses the library on every invocation, so the
  library-load wall time is measured separately (via ``--pL`` with N copies
  of a tiny PDB to solve for load and per-call analysis) and subtracted
  from each per-PDB measurement.

For each structure we record times in five configurations:
    py_numpy, py_numba, cpp                         (native_only=False)
    py_numpy_native, py_numba_native                (native_only=True; cpp lacks
                                                     this mode and is omitted)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pyconfind import analyze
from pyconfind.rotlib import load_library


def _measure_cpp_load(cpp: Path, rlib: Path, tiny_pdb: Path) -> float:
    """Estimate the C++ library-load wall time (seconds).

    Times ``confind --p tiny.pdb`` (== load + a_small) and ``confind --pL`` with
    five copies of the same tiny PDB (== load + 5*a_small). Solving the pair
    gives a_small and the load.
    """
    t0 = time.perf_counter()
    subprocess.run(
        [str(cpp), "--p", str(tiny_pdb), "--rLib", str(rlib),
         "--o", "/tmp/_cpp_load_probe.cont"],
        check=True, capture_output=True,
    )
    t1 = time.perf_counter() - t0

    Path("/tmp/_pl_load.txt").write_text("\n".join([str(tiny_pdb)] * 5) + "\n")
    Path("/tmp/_ol_load.txt").write_text(
        "\n".join(f"/tmp/_cpp_load_probe_{i}.cont" for i in range(5)) + "\n"
    )
    t0 = time.perf_counter()
    subprocess.run(
        [str(cpp), "--pL", "/tmp/_pl_load.txt", "--rLib", str(rlib),
         "--oL", "/tmp/_ol_load.txt"],
        check=True, capture_output=True,
    )
    t5 = time.perf_counter() - t0

    a_small = (t5 - t1) / 4
    return t1 - a_small


def _time_cpp(cpp: Path, rlib: Path, pdb: Path, load_offset: float) -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [str(cpp), "--p", str(pdb), "--rLib", str(rlib),
         "--o", "/tmp/_cpp_bench.cont"],
        check=True, capture_output=True,
    )
    return max(0.0, time.perf_counter() - t0 - load_offset)


def _time_py(pdb: Path, library, **kw) -> float:
    t0 = time.perf_counter()
    analyze(pdb, rotamer_library=library, **kw)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rLib", required=True, type=Path)
    ap.add_argument("--cpp", required=True, type=Path)
    ap.add_argument("--tiny", required=True, type=Path,
                    help="Tiny PDB used to estimate C++ library-load time.")
    ap.add_argument("--out", default=Path("docs/timing_results.json"), type=Path)
    ap.add_argument("pdbs", nargs="+", type=Path)
    args = ap.parse_args()

    print("Loading pyconfind rotamer library ...", flush=True)
    library = load_library(args.rLib)

    print("Measuring C++ library-load overhead ...", flush=True)
    cpp_load = _measure_cpp_load(args.cpp, args.rLib, args.tiny)
    print(f"  cpp library-load ~ {cpp_load:.2f}s (will be subtracted)\n", flush=True)

    # Warm up Numba JIT with a tiny call.
    print("Warming up Numba JIT ...", flush=True)
    analyze(args.tiny, rotamer_library=library, backend="numba")

    rows = []
    print(f"{'pdb':<22} {'nres':>4} {'py':>7} {'nb':>7} {'cpp':>7} {'py_n':>6} {'nb_n':>6}")
    for pdb in args.pdbs:
        a = analyze(pdb, rotamer_library=library, backend="numba")
        nres = len(a.positions)
        py = _time_py(pdb, library, backend="python")
        nb = _time_py(pdb, library, backend="numba")
        cpp = _time_cpp(args.cpp, args.rLib, pdb, cpp_load)
        pyn = _time_py(pdb, library, backend="python", native_only=True)
        nbn = _time_py(pdb, library, backend="numba", native_only=True)
        rows.append({
            "pdb": pdb.stem, "nres": nres,
            "py_numpy": py, "py_numba": nb, "cpp": cpp,
            "py_numpy_native": pyn, "py_numba_native": nbn,
        })
        print(f"{pdb.stem:<22} {nres:>4} {py:>6.1f}s {nb:>6.1f}s "
              f"{cpp:>6.1f}s {pyn:>5.2f}s {nbn:>5.2f}s", flush=True)
        # Save incrementally so a crash mid-batch doesn't lose work.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"cpp_load_seconds": cpp_load, "rows": rows}, indent=2,
        ))

    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
