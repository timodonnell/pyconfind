#!/usr/bin/env python3
"""Benchmark pyconfind against the C++ reference binary.

Times both implementations on a list of PDB files and reports the ratio.
The C++ binary is loaded by ``scripts/build-reference.sh``; the rotamer
library directory must contain ``EBL.out`` + ``BEBL.out`` (Dunbrack 2010
MSL format).

Usage::

    python scripts/benchmark.py \\
        --rLib original-source/confind-msl/rotlibs \\
        --cpp-binary original-source/confind-msl/mslib/bin/confind \\
        original-source/confind-msl/mslib/exampleFiles/example0000.pdb \\
        original-source/confind-msl/mslib/exampleFiles/example0002.pdb

The Python timing splits out the one-time rotamer-library load from the
per-PDB analysis so that batch workloads (where the library is amortized)
can be evaluated separately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pyconfind import analyze
from pyconfind.rotlib import load_library


def _time_python(pdb: Path, library, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        analyze(pdb, rotamer_library=library)
        times.append(time.perf_counter() - t0)
    return min(times)


def _time_cpp(binary: Path, pdb: Path, rotlib: Path, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        subprocess.run(
            [str(binary), "--p", str(pdb), "--rLib", str(rotlib),
             "--o", "/tmp/pyconfind_bench.cont"],
            check=True, capture_output=True,
        )
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rLib", required=True, type=Path,
                        help="Rotamer library directory.")
    parser.add_argument("--cpp-binary", type=Path, default=None,
                        help="Path to the reference C++ binary. If omitted, "
                             "only pyconfind is timed.")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of timing runs per PDB (best is reported).")
    parser.add_argument("pdbs", nargs="+", type=Path, help="PDB files to benchmark.")
    args = parser.parse_args()

    t0 = time.perf_counter()
    library = load_library(args.rLib)
    library_load_time = time.perf_counter() - t0
    print(f"Python rotamer library load: {library_load_time:.2f}s")

    header = f"{'PDB':<32}  {'py (no lib)':>12}  {'py +lib':>10}"
    if args.cpp_binary is not None:
        header += f"  {'C++':>8}  {'py/C++':>7}"
    print(header)
    print("-" * len(header))

    for pdb in args.pdbs:
        py_time = _time_python(pdb, library, args.repeat)
        line = f"{pdb.name:<32}  {py_time:>10.3f}s  {py_time + library_load_time:>8.3f}s"
        if args.cpp_binary is not None:
            cpp_time = _time_cpp(args.cpp_binary, pdb, args.rLib, args.repeat)
            ratio = (py_time + library_load_time) / cpp_time
            line += f"  {cpp_time:>6.3f}s  {ratio:>6.2f}x"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
