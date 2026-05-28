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
| example0000          |        3 |      0.04 s |  3.44 s | 7.30 s |          ~180×* |        2.1×  |
| example0002          |       28 |      1.22 s |  4.62 s | 9.85 s |           8.1×  |        2.1×  |
| 1UBQ                 |       76 |      4.77 s |  8.18 s | 18.2 s |           3.8×  |        2.2×  |
| AF-A1L3X0-F1         |     ~470 |     16.50 s | 19.92 s | 51.0 s |           3.1×  |        2.6×  |
| AF-O95571-F1         |     ~470 |     15.87 s | 19.28 s | 49.0 s |           3.1×  |        2.5×  |

\* The tiny example is dominated by fixed C++ startup/library-parse cost, so
the per-PDB ratio is not meaningful there.

## Where the time goes

The dominant cost is the per-pair contact-degree computation, which scales
~O(N²) in residue count (mitigated by the CA-distance cutoff). After
vectorizing the inner atom-neighbor loop with `cKDTree.sparse_distance_matrix`
+ NumPy scatter-adds (replacing a ~30M-iteration Python loop), the contact
step on 1UBQ dropped from 13.7 s to 4.8 s (2.85×).

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
