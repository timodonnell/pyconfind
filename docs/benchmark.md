# Performance

pyconfind is pure Python + NumPy + SciPy `cKDTree` — no Numba or Cython — yet
runs faster than the hand-tuned C++ `confind` binary.

## Numbers

Measured with `scripts/benchmark.py` (best of 3 runs each) on this machine.
"py (no lib)" is the per-structure analysis with the rotamer library already
loaded (the realistic batch case); "py +lib" adds the one-time ~3.4 s library
load that a cold single-shot run pays. The C++ binary re-parses the library on
every invocation.

| Structure            | Residues | py (no lib) | py +lib |    C++ | per-PDB speedup | cold speedup |
|----------------------|---------:|------------:|--------:|-------:|----------------:|-------------:|
| example0002          |       28 |      1.14 s |  4.60 s | 9.85 s |           8.7×  |        2.1×  |
| 1UBQ                 |       76 |      4.44 s |  7.90 s | 18.2 s |           4.1×  |        2.3×  |
| AF-A1L3X0-F1         |     ~470 |     14.69 s | 18.15 s | 50.9 s |           3.5×  |        2.8×  |

The one-time rotamer-library load is ~3.4 s. For the realistic batch case
(many structures, library loaded once) the per-structure speedup is the
relevant figure: **3.5-8.7×**.

## Where the time goes

The dominant cost is the per-pair contact-degree computation, which scales
~O(N²) in residue count (mitigated by the CA-distance cutoff). Two
optimizations brought it to the current state:

1. Vectorized the inner atom-neighbor loop with
   `cKDTree.sparse_distance_matrix` + NumPy scatter-adds, replacing a
   ~30M-iteration Python loop (1UBQ contact step: 13.7 s → 4.8 s).
2. Hoisted per-position constants (bounding boxes, weight sums) out of the
   per-pair inner loop, and replaced `np.cross` in the IC builder with direct
   component arithmetic (another ~7-8% on large structures).

The remaining hot spots are all inside NumPy/SciPy C code (KD-tree queries,
`np.add.at`, `np.unique`), so further large wins would need Numba/Cython or a
smarter pair-pruning scheme — not yet implemented.

## Reproduce

```bash
python scripts/benchmark.py \
    --rLib original-source/confind-msl/rotlibs \
    --cpp-binary original-source/confind-msl/mslib/bin/confind \
    <structure.pdb> [<structure2.pdb> ...]
```
