# pyconfind

A modern Python implementation of [confind](https://grigoryanlab.org/confind/) —
the rotamer-based protein side-chain contact-degree analysis introduced in
Zheng & Grigoryan's work on tertiary structural motifs.

The Python output is **byte-for-byte identical** to the upstream C++ binary
on **199 of 200** real structures (100 from the PDB + 100 from the AlphaFold
DB; see [docs/stress_test_results.md](docs/stress_test_results.md)) and on
all 11 example structures shipped with the original codebase. The single
mismatch in the stress test is one freedom row out of ~437,000 output lines.

pyconfind is also slightly faster than the C++ binary: 1.23× speedup
overall on the real-structure stress test (1.18× PDB, 1.28× AFDB).

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

CLI (matches the original `confind` flag names, so existing pipelines drop in):

```bash
pyconfind --p input.pdb --rLib path/to/rotlibs --o out.cont
# Modern structured output:
pyconfind --p input.pdb --rLib path/to/rotlibs --json --o out.json
# Only consider the native AA at each position (no AA substitution):
pyconfind --p input.pdb --rLib path/to/rotlibs --native-only --o out.cont
```

Library API:

```python
from pyconfind import analyze, format_confind_text

result = analyze("input.pdb", rotamer_library="path/to/rotlibs")
print(format_confind_text(result.positions, result.report))

# Inspect raw contacts:
for c in result.report.contacts:
    pi, pj = result.positions[c.pos_i], result.positions[c.pos_j]
    print(f"{pi.position.chain},{pi.position.resnum} <-> "
          f"{pj.position.chain},{pj.position.resnum}: degree={c.degree}")
```

## Rotamer libraries

Out of the box, pyconfind supports the Dunbrack 2010 MSL-format library that
ships with the upstream confind source (`EBL.out` + `BEBL.out`). Point
`--rLib` at a directory containing both files (backbone-dependent) or at a
single EBL.out-style file (backbone-independent).

Modern Dunbrack and Richardson-style libraries are next on the roadmap.

## Native-only mode (extension over the C++ binary)

The original C++ confind substitutes in all 18 non-Gly/Pro amino acids at
every position and computes contact degree across the full rotamer space.
pyconfind adds `--native-only`: at each position, only place rotamers of the
native amino acid (but still consider every rotamer of that AA). Useful when
you want a contact-degree estimate that holds the sequence fixed.

## Validation

The C++ reference binary is built from the upstream tarball by:

```bash
scripts/build-reference.sh
```

The byte-identity tests then compare pyconfind's output against the C++
output on every example PDB. To run them yourself:

```bash
pytest tests/
```

## References

* "Sequence statistics of tertiary structural motifs reflect protein stability",
  F. Zheng, G. Grigoryan, PLoS ONE, 12(5): e0178272, 2017.

* "Tertiary Structural Propensities Reveal Fundamental Sequence/Structure
  Relationships", F. Zheng, J. Zhang, G. Grigoryan, Structure, 23(5):
  961-971, 2015.
