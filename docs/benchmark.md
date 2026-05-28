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

Per-structure analysis (rotamer library already loaded — the realistic batch
case), best of several runs.

| Structure    | Residues | py backend | numba backend | numba speedup | vs C++ (per-structure) |
|--------------|---------:|-----------:|--------------:|--------------:|-----------------------:|
| example0002  |       28 |    1.13 s  |       0.55 s  |        2.1×   |          ~18×          |
| 1UBQ         |       76 |    4.44 s  |       1.76 s  |        2.5×   |          ~10×          |
| AF-A1L3X0-F1 |     ~470 |   14.70 s  |       6.04 s  |        2.4×   |           ~8×          |
| 1CBW         |      594 |   35.68 s  |      12.32 s  |        2.9×   |           ~8×          |

The one-time rotamer-library load (~3.4 s) is excluded — it is amortized when
processing a batch. The C++ binary re-parses the library on every invocation,
so cold single-shot speedups are larger still.

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
python scripts/benchmark.py \
    --rLib original-source/confind-msl/rotlibs \
    --cpp-binary original-source/confind-msl/mslib/bin/confind \
    <structure.pdb> [<structure2.pdb> ...]
```
