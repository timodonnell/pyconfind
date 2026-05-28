# Real-world stress test: pyconfind vs. C++ confind

Tested pyconfind against the reference C++ binary on 200 real structures —
100 from the RCSB PDB (single-chain X-ray entries, 50-200 residues) and
100 from the AlphaFold DB (human reviewed UniProt, 50-300 residue sequences).

## Results

|              | Structures | Byte-identical | Speedup |
|--------------|-----------:|---------------:|--------:|
| **PDB**      |        100 |            100 | 1.17× |
| **AFDB**     |        100 |            100 | 1.27× |
| **Combined** |        200 |        **200** | 1.22× |

Total runtime: 6221 s (C++) vs. 5104 s (pyconfind). Every output row —
contact, sumcond, percont, crwdnes, freedom, SEQUENCE — is byte-for-byte
identical to the reference binary across all 200 structures.

## A formerly-tricky case (now fixed)

An earlier run had a single `freedom` mismatch at `AF-O15116-F1` position
`A,68` (a GLY residue): 0.001887 vs. the C++'s 0.001334. The cause was a
zero-weight rotamer. MSL's `contactProbability` accumulates
collision-probability mass into `cp[i] += p2` even when the i-side rotamer
weight is zero (and a few EBL rotamer entries have weight exactly 0.0). Our
contact loop was early-returning when `sum(weights_i) * sum(weights_j) == 0`,
skipping that accumulation, so the lone surviving zero-weight rotamer failed
to register in the freedom counter. Removing the early return (while still
returning NaN for the 0/0 contact degree) brought it into agreement.

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
