#!/usr/bin/env python3
"""Stress test: compare pyconfind to the reference C++ binary on real PDB +
AlphaFold DB structures.

Downloads N PDB IDs and N AFDB UniProt-derived structures, runs both
implementations on each, and reports any mismatches in the output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = Request(url, headers={"User-Agent": "pyconfind-stress-test/0.1"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_pdb_ids(n: int, offset: int = 0) -> list[str]:
    """Use the RCSB search API to fetch single-chain protein X-ray entries
    with 50-200 residues."""
    body = {
        "query": {
            "type": "group", "logical_operator": "and", "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                                "operator": "equals", "value": 1}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                                "operator": "range",
                                "value": {"from": 50, "to": 200}}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "exptl.method",
                                "operator": "exact_match",
                                "value": "X-RAY DIFFRACTION"}},
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": offset, "rows": n},
            "results_content_type": ["experimental"],
        },
    }
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=" + quote(json.dumps(body))
    data = json.loads(_http_get(url))
    return [r["identifier"] for r in data["result_set"][:n]]


def fetch_pdb(pdb_id: str, out_dir: Path) -> Path | None:
    out = out_dir / f"{pdb_id}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        data = _http_get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    except Exception as e:
        print(f"  fetch {pdb_id} failed: {e}", file=sys.stderr)
        return None
    out.write_bytes(data)
    return out


def fetch_uniprot_ids(n: int) -> list[str]:
    """List ``n`` human reviewed UniProt IDs in the 50-300 residue range."""
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        f"query=organism_id:9606+AND+reviewed:true+AND+length:%5B50+TO+300%5D"
        f"&format=list&size={n}"
    )
    text = _http_get(url).decode()
    return [line.strip() for line in text.splitlines() if line.strip()][:n]


def fetch_afdb(uniprot_id: str, out_dir: Path) -> Path | None:
    """Resolve the current AFDB pdbUrl and download."""
    out = out_dir / f"AF-{uniprot_id}-F1.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        meta = json.loads(_http_get(meta_url))
        pdb_url = meta[0]["pdbUrl"]
        data = _http_get(pdb_url)
    except Exception as e:
        print(f"  fetch AF {uniprot_id} failed: {e}", file=sys.stderr)
        return None
    out.write_bytes(data)
    return out


@dataclass
class Comparison:
    pdb: str
    label: str               # "PDB" or "AFDB"
    status: str              # "match" | "diff" | "error_cpp" | "error_py"
    diff_lines: int = 0
    total_lines: int = 0
    py_seconds: float = 0.0
    cpp_seconds: float = 0.0
    message: str = ""


def run_cpp(binary: Path, pdb: Path, rotlib: Path, out: Path, timeout: float) -> tuple[bool, str, float]:
    t0 = time.perf_counter()
    try:
        subprocess.run(
            [str(binary), "--p", str(pdb), "--rLib", str(rotlib), "--o", str(out)],
            check=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "cpp timeout", time.perf_counter() - t0
    except subprocess.CalledProcessError as e:
        return False, f"cpp exit {e.returncode}: {e.stderr.decode(errors='replace')[:200]}", time.perf_counter() - t0
    except Exception as e:
        return False, f"cpp error: {e}", time.perf_counter() - t0
    return True, "", time.perf_counter() - t0


def run_py(pdb: Path, library, out: Path, timeout: float) -> tuple[bool, str, float]:
    # Run in-process (library reused across calls). Timeout is enforced via
    # a wall-clock check before returning; on huge structures the call may
    # exceed it, which we tolerate but note.
    from pyconfind import analyze, format_confind_text
    t0 = time.perf_counter()
    try:
        a = analyze(pdb, rotamer_library=library)
    except Exception as e:
        return False, f"py error: {type(e).__name__}: {str(e)[:200]}", time.perf_counter() - t0
    text = format_confind_text(a.positions, a.report)
    out.write_text(text)
    return True, "", time.perf_counter() - t0


def diff_outputs(cpp_path: Path, py_path: Path) -> tuple[int, int]:
    """Return ``(diff_lines, total_lines)``."""
    cpp = cpp_path.read_text().splitlines()
    py = py_path.read_text().splitlines()
    pad = max(len(cpp), len(py))
    cpp += [""] * (pad - len(cpp))
    py += [""] * (pad - len(py))
    diff = sum(1 for a, b in zip(cpp, py, strict=True) if a != b)
    return diff, pad


def compare_one(
    pdb_path: Path,
    label: str,
    cpp_binary: Path,
    rotlib: Path,
    library,
    cpp_out_dir: Path,
    py_out_dir: Path,
    diff_dir: Path,
    timeout: float,
) -> Comparison:
    name = pdb_path.stem
    cpp_out = cpp_out_dir / f"{name}.cont"
    py_out = py_out_dir / f"{name}.cont"

    ok_cpp, msg_cpp, t_cpp = run_cpp(cpp_binary, pdb_path, rotlib, cpp_out, timeout)
    if not ok_cpp:
        return Comparison(pdb=name, label=label, status="error_cpp", message=msg_cpp, cpp_seconds=t_cpp)

    ok_py, msg_py, t_py = run_py(pdb_path, library, py_out, timeout)
    if not ok_py:
        return Comparison(pdb=name, label=label, status="error_py", message=msg_py, py_seconds=t_py, cpp_seconds=t_cpp)

    diff, total = diff_outputs(cpp_out, py_out)
    if diff == 0:
        return Comparison(pdb=name, label=label, status="match",
                          total_lines=total, py_seconds=t_py, cpp_seconds=t_cpp)
    # Save diff for inspection.
    diff_path = diff_dir / f"{name}.diff"
    with diff_path.open("w") as f:
        cpp_lines = cpp_out.read_text().splitlines()
        py_lines = py_out.read_text().splitlines()
        pad = max(len(cpp_lines), len(py_lines))
        cpp_lines += [""] * (pad - len(cpp_lines))
        py_lines += [""] * (pad - len(py_lines))
        for i, (a, b) in enumerate(zip(cpp_lines, py_lines, strict=True)):
            if a != b:
                f.write(f"L{i}:\n  cpp: {a}\n  py : {b}\n")
    return Comparison(pdb=name, label=label, status="diff",
                      diff_lines=diff, total_lines=total,
                      py_seconds=t_py, cpp_seconds=t_cpp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rLib", required=True, type=Path)
    parser.add_argument("--cpp-binary", required=True, type=Path)
    parser.add_argument("--pdb-count", type=int, default=100)
    parser.add_argument("--afdb-count", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/pyconfind_test"))
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-PDB timeout for the C++ run.")
    parser.add_argument("--pdb-offset", type=int, default=0,
                        help="Skip the first N PDB hits (to get a different sample).")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="If set, write a machine-readable summary here.")
    args = parser.parse_args()

    from pyconfind.rotlib import load_library
    library = load_library(args.rLib)
    print(f"Library loaded.", flush=True)

    pdb_dir = args.data_dir / "pdb"
    afdb_dir = args.data_dir / "afdb"
    cpp_out = args.data_dir / "cpp_out"
    py_out = args.data_dir / "py_out"
    diff_dir = args.data_dir / "diffs"
    for d in (pdb_dir, afdb_dir, cpp_out, py_out, diff_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.pdb_count} PDB IDs...", flush=True)
    pdb_ids = fetch_pdb_ids(args.pdb_count + 25, offset=args.pdb_offset)  # over-fetch for failures
    print(f"  got {len(pdb_ids)} ids", flush=True)

    print(f"Downloading PDB files...", flush=True)
    pdb_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for p in ex.map(lambda i: fetch_pdb(i, pdb_dir), pdb_ids):
            if p is not None:
                pdb_paths.append(p)
            if len(pdb_paths) >= args.pdb_count:
                break

    print(f"  {len(pdb_paths)} PDB files ready", flush=True)

    print(f"Fetching UniProt IDs...", flush=True)
    uniprot_ids = fetch_uniprot_ids(args.afdb_count + 25)
    print(f"  got {len(uniprot_ids)} ids", flush=True)

    print(f"Downloading AFDB files...", flush=True)
    afdb_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for p in ex.map(lambda u: fetch_afdb(u, afdb_dir), uniprot_ids):
            if p is not None:
                afdb_paths.append(p)
            if len(afdb_paths) >= args.afdb_count:
                break
    print(f"  {len(afdb_paths)} AFDB files ready", flush=True)

    results: list[Comparison] = []
    for label, paths in (("PDB", pdb_paths), ("AFDB", afdb_paths)):
        print(f"\nRunning {label} ({len(paths)} structures)...", flush=True)
        for i, p in enumerate(paths, 1):
            t0 = time.perf_counter()
            r = compare_one(
                p, label, args.cpp_binary, args.rLib, library,
                cpp_out, py_out, diff_dir, args.timeout,
            )
            results.append(r)
            elapsed = time.perf_counter() - t0
            tag = {"match": "✓", "diff": "✗", "error_cpp": "C", "error_py": "P"}[r.status]
            print(f"  [{i}/{len(paths)}] {tag} {r.pdb:8s} "
                  f"({r.total_lines or 0:4d}L, {r.diff_lines or 0:3d} diff, "
                  f"py {r.py_seconds:5.1f}s, cpp {r.cpp_seconds:5.1f}s, total {elapsed:5.1f}s) "
                  f"{r.message[:80]}", flush=True)

    # Summary
    print("\n" + "=" * 70)
    by_label: dict[str, dict[str, int]] = {}
    for r in results:
        by_label.setdefault(r.label, {}).setdefault(r.status, 0)
        by_label[r.label][r.status] += 1
    for label, counts in by_label.items():
        total = sum(counts.values())
        print(f"\n{label}: {total} structures")
        for status, n in sorted(counts.items()):
            print(f"  {status:12s} {n:3d}  ({100 * n / total:.1f}%)")
    # Diff-line distribution for diffs:
    diffs = [r for r in results if r.status == "diff"]
    if diffs:
        print(f"\nMismatched structures ({len(diffs)}):")
        for r in diffs[:20]:
            print(f"  {r.label} {r.pdb}: {r.diff_lines}/{r.total_lines} lines")
        print(f"\nDiffs saved to {diff_dir}/")

    if args.summary_json is not None:
        payload = {
            "results": [
                {
                    "pdb": r.pdb, "label": r.label, "status": r.status,
                    "diff_lines": r.diff_lines, "total_lines": r.total_lines,
                    "py_seconds": r.py_seconds, "cpp_seconds": r.cpp_seconds,
                    "message": r.message,
                }
                for r in results
            ],
        }
        args.summary_json.write_text(json.dumps(payload, indent=2))
        print(f"\nFull report: {args.summary_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
