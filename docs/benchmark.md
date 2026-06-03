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

### Full analysis (`native_only=False`)

| Structure    | Residues | numpy   | numba   | C++ (analysis only) | numba vs C++ |
|--------------|---------:|--------:|--------:|--------------------:|-------------:|
| AF-A1L190-F1 |       88 |  3.13 s |  1.58 s |              8.51 s |       5.4×   |
| AF-A6NNB3-F1 |      132 |  5.11 s |  2.56 s |             14.48 s |       5.6×   |
| AF-A6NI61-F1 |      221 | 10.38 s |  4.44 s |             30.46 s |       6.9×   |
| 1AB9         |      242 | 14.92 s |  4.99 s |             39.22 s |       7.8×   |
| AF-A1L3X0-F1 |      281 | 14.49 s |  5.91 s |             44.32 s |       7.6×   |
| 1C08         |      350 | 21.39 s |  7.66 s |             64.05 s |       8.4×   |
| 1B0R         |      375 | 23.20 s |  8.27 s |             69.10 s |       8.4×   |
| 1BWU         |      430 | 26.33 s |  9.11 s |             76.46 s |       8.5×   |
| 1AVG         |      442 | 27.03 s |  9.55 s |             82.95 s |       8.7×   |
| 1C04         |      488 | 26.84 s |  9.95 s |             71.93 s |       7.2×   |
| 1BQL         |      555 | 36.51 s | 12.41 s |            105.55 s |       8.5×   |

### Native-only (`native_only=True`)

| Structure    | Residues | numpy   | numba   |
|--------------|---------:|--------:|--------:|
| AF-A1L190-F1 |       88 | 0.124 s | 0.088 s |
| AF-A6NNB3-F1 |      132 | 0.143 s | 0.097 s |
| AF-A6NI61-F1 |      221 | 0.310 s | 0.173 s |
| 1AB9         |      242 | 0.355 s | 0.202 s |
| AF-A1L3X0-F1 |      281 | 0.450 s | 0.264 s |
| 1C08         |      350 | 0.585 s | 0.311 s |
| 1B0R         |      375 | 0.696 s | 0.398 s |
| 1BWU         |      430 | 0.643 s | 0.361 s |
| 1AVG         |      442 | 0.840 s | 0.452 s |
| 1C04         |      488 | 0.704 s | 0.429 s |
| 1BQL         |      555 | 0.901 s | 0.531 s |

Median speedups (numba backend, library pre-loaded everywhere):

* **numba vs C++** (full mode): 7.8×
* **numba vs numpy** (full mode): 2.8×
* **`native_only=True`** vs full mode (numba): ~23× faster again — every
  structure in this set finishes in **under 0.55 s**, smaller ones in ~0.1 s.

The phi/psi numba kernel (added in `b5ba233`) shaves roughly **+14% median**
off native-only numba times (range +5..+21%), and ~2% off full-mode times
(where rotamer building dominates). Raw data:
[docs/timing_results.json](timing_results.json).

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

A second numba kernel (`structure._dihedrals_kernel`) batches the
phi/psi/omega computation across all positions, replacing what used to be a
per-position `np.cross`/`np.dot` loop. This matters most for `native_only=True`
runs, where rotamer building is cheap and the dihedral pass would otherwise be
a sizable fraction of the call: 5TRU (542 residues, `native_only=True`) drops
from ~0.71 s to ~0.48 s per call. The dihedral arithmetic is hand-scalarized
over the length-3 vectors so it is bit-equivalent to the original — the
byte-identity goldens are the canary.

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
