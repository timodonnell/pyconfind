# Real-world stress test: pyconfind vs. C++ confind

Tested pyconfind against the reference C++ binary on 200 real structures —
100 from the RCSB PDB (single-chain X-ray entries, 50-200 residues) and
100 from the AlphaFold DB (human reviewed UniProt, 50-300 residue sequences).

## Results

|              | Structures | Byte-identical | Diff lines / output | Speedup |
|--------------|-----------:|---------------:|--------------------:|--------:|
| **PDB**      |        100 |            100 |              0 / 224k+ | 1.18× |
| **AFDB**     |        100 |             99 |              1 / 213k+ | 1.28× |
| **Combined** |        200 |            199 |              1 / 437k+ | 1.23× |

Total runtime: 6222 s (C++) vs. 5069 s (pyconfind).

## The single mismatch

`AF-O15116-F1` position `A,68` (a GLY residue). The diff is a single
`freedom` value:

```
cpp: freedom    A,68    0.001334    GLY
py : freedom    A,68    0.001887    GLY
```

Ratio: √2. The `freedom` metric counts rotamers below two collision-
probability thresholds (0.5 and 2.0 — see confind.cpp:797-826) and rolls
them into `sqrt((n1² + n2²) / 2) / orig_num_rotamers`. One rotamer's
collision-probability mass lands on opposite sides of the `cp/100 < 0.5`
threshold between the two implementations — a discrete-threshold
floating-point boundary effect on a single rotamer out of thousands, not
a fundamental disagreement (the underlying contact-pair set is unchanged).

## Reproduce

```bash
# Build the reference C++ binary once.
scripts/build-reference.sh

# Run the comparison.
python scripts/stress_test.py \
    --rLib original-source/confind-msl/rotlibs \
    --cpp-binary original-source/confind-msl/mslib/bin/confind \
    --pdb-count 100 --afdb-count 100 \
    --summary-json /tmp/summary.json

# Generate the plot.
python scripts/plot_stress_test.py --summary /tmp/summary.json \
    --out docs/stress_test.png
```

![runtime comparison](stress_test.png)
