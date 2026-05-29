# pyconfind

[![CI](https://github.com/timodonnell/pyconfind/actions/workflows/ci.yml/badge.svg)](https://github.com/timodonnell/pyconfind/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyconfind.svg)](https://pypi.org/project/pyconfind/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timodonnell/pyconfind/blob/main/examples/pyconfind_demo.ipynb)

A modern Python implementation of [confind](https://grigoryanlab.org/confind/) —
the rotamer-based protein side-chain contact-degree analysis introduced in
Zheng & Grigoryan's work on tertiary structural motifs.

The Python output is **byte-for-byte identical** to the upstream C++ binary
on **248 of 253** real structures tested (100 single-chain PDB + 100 AlphaFold
DB + 50 multi-chain + 3 high-resolution; see
[docs/stress_test_results.md](docs/stress_test_results.md)), plus a further 100
RCSB entries cross-checked as both PDB and mmCIF. The 5 exceptions are
insertion-code structures where the C++ ordering relies on undefined behavior
(documented). The test suite runs against real PDB/mmCIF structures with
committed C++-reference contact maps.

pyconfind is also faster than the C++ binary, with two interchangeable
contact-degree backends (both byte-identical to the reference):

* a pure NumPy/SciPy reference, which on its own already beats the C++ binary;
* an optional **Numba** JIT/multi-threaded backend (`pip install pyconfind[fast]`)
  that is ~2-3× faster again.

With the Numba backend and the rotamer library amortized across a batch, the
per-structure analysis is **~8-18× faster** than the C++ binary.

![runtime vs sequence length](docs/timing_vs_length.png)

Runtime scales sub-quadratically with sequence length (the CA-distance cutoff
bounds each residue's neighbor count). See [docs/benchmark.md](docs/benchmark.md)
for details.

## Install

```bash
pip install pyconfind            # pure-Python reference backend
pip install "pyconfind[fast]"    # + Numba JIT/multi-threaded backend
```

From source (for development):

```bash
pip install -e ".[dev]"          # editable install with test/lint tooling
```

## Example notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timodonnell/pyconfind/blob/main/examples/pyconfind_demo.ipynb)

[`examples/pyconfind_demo.ipynb`](examples/pyconfind_demo.ipynb) is a runnable
walkthrough (install → fetch a PDB → analyze via the library API → visualize a
contact map, per-residue scores, and a 3D structure colored by contact degree).
Click the badge to run it on a free Colab CPU runtime.

## Quick start

CLI (matches the original `confind` flag names, so existing pipelines drop in):

```bash
pyconfind --p input.pdb --rLib path/to/rotlibs --o out.cont
# Inputs may be PDB or mmCIF (format auto-detected via gemmi):
pyconfind --p input.cif --rLib path/to/rotlibs --o out.cont
# Modern structured output:
pyconfind --p input.pdb --rLib path/to/rotlibs --json --o out.json
# Only consider the native AA at each position (no AA substitution):
pyconfind --p input.pdb --rLib path/to/rotlibs --native-only --o out.cont
# Restrict the computed/output residues (MSL selection language):
pyconfind --p input.pdb --rLib path/to/rotlibs --sel "chain A AND resi 20-60" --o out.cont
# Pre-select part of the structure before anything runs:
pyconfind --p input.pdb --rLib path/to/rotlibs --psel "NAME CA WITHIN 25 OF CHAIN A" --o out.cont
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
