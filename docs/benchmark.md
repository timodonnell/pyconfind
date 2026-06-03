# Performance

pyconfind ships two interchangeable contact-degree backends:

* **`python`** — pure NumPy + SciPy `cKDTree`, the reference implementation.
* **`numba`** — a JIT-compiled, multi-threaded kernel (the default when Numba
  is installed; `pip install pyconfind[fast]`).

Both produce results identical to ~1e-15 (far below the 1e-6 print precision),
so output stays byte-identical to the C++ reference either way. The pure-Python
backend already beats the hand-tuned C++ binary; the Numba backend is ~2-3×
faster again on top of that.

## Numbers

Per-structure analysis time vs. sequence length, rotamer library pre-loaded
and **excluded from every measurement** (the realistic batch case). Eleven
structures spanning ~88-555 residues, measured on the same machine. C++
library-load wall time (~6.6 s on the bench machine) was measured separately
via ``confind --pL`` and subtracted from each C++ data point so it isn't
counted twice.

| Structure    | Residues | numpy backend | numba backend | C++ (analysis only) | numba vs C++ |
|--------------|---------:|--------------:|--------------:|--------------------:|-------------:|
| AF-A1L190-F1 |       88 |      3.11 s   |      1.62 s   |              8.46 s |       5.2×   |
| AF-A6NNB3-F1 |      132 |      5.12 s   |      2.64 s   |             14.37 s |       5.4×   |
| AF-A6NI61-F1 |      221 |     10.54 s   |      4.54 s   |             30.42 s |       6.7×   |
| 1AB9         |      242 |     14.42 s   |      5.04 s   |             39.21 s |       7.8×   |
| AF-A1L3X0-F1 |      281 |     14.71 s   |      6.13 s   |             44.74 s |       7.3×   |
| 1C08         |      350 |     22.32 s   |      7.73 s   |             64.34 s |       8.3×   |
| 1B0R         |      375 |     23.34 s   |      8.35 s   |             69.00 s |       8.3×   |
| 1BWU         |      430 |     26.66 s   |      9.18 s   |             76.38 s |       8.3×   |
| 1AVG         |      442 |     27.92 s   |      9.78 s   |             80.89 s |       8.3×   |
| 1C04         |      488 |     26.93 s   |     10.05 s   |             71.73 s |       7.1×   |
| 1BQL         |      555 |     36.23 s   |     12.73 s   |            105.77 s |       8.3×   |

Median speedups (numba backend, library pre-loaded everywhere):

* **numba vs C++**: 7.8×
* **numba vs numpy**: 2.8×
* **`native_only=True`** vs full mode (numba): ~21× faster again — sub-second
  for everything in this set.

Raw data: [docs/timing_results.json](timing_results.json).

## Where the time goes

The two costs are rotamer building (IC placement + backbone-clash pruning) and
the per-pair contact-degree computation. The latter scales ~O(N²) in residue
count (mitigated by the CA-distance cutoff).

The **pure-Python contact path** was first made fast by:

1. Vectorizing the inner atom-neighbor loop with
   `cKDTree.sparse_distance_matrix` + NumPy scatter-adds, replacing a
   ~30M-iteration Python loop (1UBQ contact step: 13.7 s → 4.8 s).
2. Hoisting per-position constants (bounding boxes, weight sums) out of the
   per-pair loop and replacing `np.cross` in the IC builder with direct
   component arithmetic.

The **Numba backend** (`contacts_numba.py`) then replaces the contact
computation with a JIT-compiled, multi-threaded kernel — ~4.6× faster than the
already-optimized Python contact step (470-residue structure: 10.9 s → 2.4 s),
making rotamer building the new dominant cost.

### Why rotamer building is not (yet) Numba-accelerated

It would be the next target, but it is deliberately left in exact NumPy. The
backbone-clash prune compares atom distances against a hard 2.0 Å threshold; a
re-implementation that computes those distances even slightly differently
could flip a borderline clash decision, which would change which rotamers
survive and cascade into different contacts. That would jeopardize the
byte-identity guarantee. The contact-degree backend is safe to accelerate
because floating-point reordering there only perturbs the 6th+ decimal of a
degree, never a discrete decision.

## Reproduce

```bash
python scripts/benchmark_v2.py \
    --rLib original-source/confind-msl/rotlibs \
    --cpp  original-source/confind-msl/mslib/bin/confind \
    --tiny tests/data/structures/1CRN.pdb \
    --out  docs/timing_results.json \
    <structure.pdb> [<structure2.pdb> ...]

python scripts/plot_timing_vs_length.py
```

``--tiny`` is the small reference PDB used to measure the C++ library-load
overhead (subtracted from each C++ data point so the comparison is fair).
