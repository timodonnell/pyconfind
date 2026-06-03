# Performance

pyconfind ships two interchangeable contact-degree backends:

* **`python`** — pure NumPy + SciPy `cKDTree`, the reference implementation.
* **`numba`** — a JIT-compiled, multi-threaded kernel (the default when Numba
  is installed; `pip install pyconfind[fast]`).

Both produce results identical to ~1e-15 (far below the 1e-6 print precision),
so output stays byte-identical to the C++ reference either way. The pure-Python
backend already beats the hand-tuned C++ binary; the Numba backend is faster
again on top of that.

When `pyconfind[fast]` is installed, three numba kernels run regardless of
the `backend=` choice — they accelerate steps that are not contact-degree
computation and are hand-scalarized to be bit-equivalent to the original
NumPy code:

* `structure._dihedrals_kernel` — batched phi/psi/omega computation
* `geometry._place_batch_kernel` — IC placement of all rotamers per call
* `contacts_numba` — the contact-degree kernel itself (this is what
  `backend="numba"` selects; `backend="python"` falls back to NumPy/SciPy
  for the contact step only)

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
| AF-A1L190-F1 |       88 |  2.74 s |  1.18 s |              8.62 s |       7.3×   |
| AF-A6NNB3-F1 |      132 |  4.44 s |  1.93 s |             14.41 s |       7.5×   |
| AF-A6NI61-F1 |      221 |  9.32 s |  3.37 s |             30.46 s |       9.0×   |
| 1AB9         |      242 | 13.10 s |  3.85 s |             39.31 s |      10.2×   |
| AF-A1L3X0-F1 |      281 | 14.00 s |  4.49 s |             44.55 s |       9.9×   |
| 1C08         |      350 | 19.66 s |  5.96 s |             64.20 s |      10.8×   |
| 1B0R         |      375 | 21.93 s |  6.42 s |             69.33 s |      10.8×   |
| 1BWU         |      430 | 24.92 s |  6.99 s |             76.40 s |      10.9×   |
| 1AVG         |      442 | 25.42 s |  7.35 s |             81.05 s |      11.0×   |
| 1C04         |      488 | 25.13 s |  7.44 s |             71.94 s |       9.7×   |
| 1BQL         |      555 | 33.16 s |  9.76 s |            105.74 s |      10.8×   |

### Native-only (`native_only=True`)

| Structure    | Residues | numpy   | numba   |
|--------------|---------:|--------:|--------:|
| AF-A1L190-F1 |       88 | 0.098 s | 0.066 s |
| AF-A6NNB3-F1 |      132 | 0.144 s | 0.069 s |
| AF-A6NI61-F1 |      221 | 0.263 s | 0.127 s |
| 1AB9         |      242 | 0.310 s | 0.145 s |
| AF-A1L3X0-F1 |      281 | 0.390 s | 0.239 s |
| 1C08         |      350 | 0.515 s | 0.234 s |
| 1B0R         |      375 | 0.651 s | 0.303 s |
| 1BWU         |      430 | 0.554 s | 0.262 s |
| 1AVG         |      442 | 0.736 s | 0.347 s |
| 1C04         |      488 | 0.600 s | 0.322 s |
| 1BQL         |      555 | 0.784 s | 0.359 s |

Median speedups (numba backend, library pre-loaded everywhere):

* **numba vs C++** (full mode): **10.2×** (range 7.3-11.0×)
* **numba vs numpy** (full mode): **3.4×**
* **`native_only=True`** vs full mode (numba): ~26× faster again — every
  structure in this set finishes in **under 0.36 s**, smaller ones in ~0.07 s.

The IC numba kernel (this branch) takes the full-mode numba backend from
~7.8× → ~10.2× over C++ and lifts native-only numba by another ~+25% median
on top of the phi/psi gains. Raw data:
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
a sizable fraction of the call. The dihedral arithmetic is hand-scalarized
over the length-3 vectors so it is bit-equivalent to the original — the
byte-identity goldens are the canary.

A third numba kernel (`geometry._place_batch_kernel`) handles the inner
**IC placement** for every rotamer at every position. The backbone-clash
prune then compares the placed coordinates against a hard 2.0 Å threshold;
because that threshold turns infinitesimal FP perturbations into a discrete
yes/no decision, this kernel was held back until we could verify it
preserves byte-identity. The kernel is hand-scalarized over the length-3
vectors and uses `fastmath=False`, which makes its arithmetic bit-equivalent
to the original NumPy form. Validation:

* All in-repo byte-identity goldens (1CRN, 1UBQ, 1EJG, 5TRU bio-assembly 1)
  pass.
* A wider sweep over 77 real structures (mixed PDB + AlphaFold DB,
  88-876 residues) shows **77/77 outputs byte-identical to the pure-NumPy
  IC builder on `main`** — no new mismatches against the C++ reference,
  even on the insertion-code structures documented in
  [`docs/stress_test_results.md`](stress_test_results.md).

The contact-degree backend remains safe to accelerate for a different reason:
floating-point reordering there only perturbs the 6th+ decimal of a degree,
never a discrete decision.

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
