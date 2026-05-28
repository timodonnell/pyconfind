"""Atom-selection language compatible with MSL's ``AtomSelection``.

Supports the subset of the MSL selection grammar that confind's ``--psel`` and
``--sel`` flags rely on:

Atomic conditions (case-insensitive keywords):

* ``ALL``
* ``NAME <a>[+<b>...]``   — atom name(s), ``+`` separates alternatives
* ``RESI <spec>`` / ``RESID <spec>`` — residue number(s): ``+`` lists,
  ``-`` ranges (``6-9``), or a positionId with insertion code (``37A``)
* ``RESN <a>[+<b>...]``   — residue name(s)
* ``CHAIN <a>[+<b>...]``  — chain id(s)
* ``HASCRD`` / ``HASCOOR`` [bool] — has coordinates

Boolean combination with ``NOT`` > ``AND`` > {``OR``, ``XOR``} precedence and
parentheses (verified against the C++ binary).

A ``WITHIN <dist> OF <selection>`` clause is also supported: the atoms matching
the left-hand selection that lie within ``dist`` Å of any atom matching the
right-hand selection.

:func:`select_residue_mask` implements MSL's "by-residue-CA" rule used by
confind: a residue is selected iff its ``CA`` atom satisfies the selection.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial import cKDTree

from .pdb import Atoms

_TOKEN_RE = re.compile(r"\(|\)|\s+|[^()\s]+")


def _tokenize(s: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(s):
        tok = m.group(0)
        if tok.strip() == "":
            continue
        out.append(tok)
    return out


def _atomic_mask(tokens: list[str], atoms: Atoms) -> np.ndarray:
    """Evaluate a single atomic condition (already split into tokens)."""
    if not tokens:
        raise ValueError("empty selection condition")
    key = tokens[0].upper()
    n = len(atoms)
    if key == "ALL" and len(tokens) == 1:
        return np.ones(n, dtype=bool)
    if key == "NAME" and len(tokens) == 2:
        vals = tokens[1].split("+")
        return np.isin(atoms.name, vals)
    if key in ("RESI", "RESID") and len(tokens) == 2:
        return _resi_mask(tokens[1], atoms)
    if key == "RESN" and len(tokens) == 2:
        vals = tokens[1].split("+")
        return np.isin(atoms.resname, vals)
    if key == "CHAIN" and len(tokens) == 2:
        vals = tokens[1].split("+")
        return np.isin(atoms.chain, vals)
    if key in ("HASCRD", "HASCOOR"):
        # We only retain atoms that had coordinates, so this is always true.
        want = True
        if len(tokens) == 2:
            want = tokens[1].upper() in ("TRUE", "1", "T", "YES")
        return np.full(n, want, dtype=bool)
    raise ValueError(f"unrecognized selection condition: {' '.join(tokens)}")


def _resi_mask(spec: str, atoms: Atoms) -> np.ndarray:
    """Match the RESI/RESID spec: ``+`` lists, ``-`` ranges, ``37A`` positionIds."""
    mask = np.zeros(len(atoms), dtype=bool)
    # positionId-with-icode form: resnum + optional insertion code.
    pos_id = np.array(
        [f"{int(r)}{ic}" for r, ic in zip(atoms.resnum, atoms.icode, strict=True)],
        dtype=object,
    )
    for val in spec.split("+"):
        rng = val.split("-")
        if len(rng) == 2 and rng[0].lstrip("-").isdigit() and rng[1].lstrip("-").isdigit():
            start, end = int(rng[0]), int(rng[1])
            mask |= (atoms.resnum >= start) & (atoms.resnum <= end)
        else:
            mask |= pos_id == val
    return mask


# --- boolean expression parser (recursive descent) -------------------------
# Grammar (precedence low→high):
#   or_expr   := and_expr (("OR"|"XOR") and_expr)*
#   and_expr  := not_expr ("AND" not_expr)*
#   not_expr  := "NOT" not_expr | atom
#   atom      := "(" or_expr ")" | condition
# A condition is a maximal run of tokens that are not operators/parens.


_OPERATORS = {"AND", "OR", "XOR", "NOT"}


class _Parser:
    def __init__(self, tokens: list[str], atoms: Atoms):
        self.tokens = tokens
        self.pos = 0
        self.atoms = atoms

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> np.ndarray:
        mask = self._or_expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"trailing tokens in selection: {self.tokens[self.pos:]}")
        return mask

    def _or_expr(self) -> np.ndarray:
        mask = self._and_expr()
        while (tok := self._peek()) is not None and tok.upper() in ("OR", "XOR"):
            op = self._next().upper()
            rhs = self._and_expr()
            mask = (mask != rhs) if op == "XOR" else (mask | rhs)
        return mask

    def _and_expr(self) -> np.ndarray:
        mask = self._not_expr()
        while (tok := self._peek()) is not None and tok.upper() == "AND":
            self._next()
            mask = mask & self._not_expr()
        return mask

    def _not_expr(self) -> np.ndarray:
        tok = self._peek()
        if tok is not None and tok.upper() == "NOT":
            self._next()
            return ~self._not_expr()
        return self._atom()

    def _atom(self) -> np.ndarray:
        tok = self._peek()
        if tok == "(":
            self._next()
            mask = self._or_expr()
            if self._peek() != ")":
                raise ValueError("unbalanced parentheses in selection")
            self._next()
            return mask
        # Gather a maximal run of non-operator, non-paren tokens.
        cond: list[str] = []
        while (t := self._peek()) is not None and t != "(" and t != ")" and t.upper() not in _OPERATORS:
            cond.append(self._next())
        if not cond:
            raise ValueError(f"expected a condition near {tok!r}")
        return _atomic_mask(cond, self.atoms)


def _split_within(tokens: list[str]) -> tuple[list[str], float, list[str]] | None:
    """If a top-level ``WITHIN <d> OF`` exists, split into (left, dist, right)."""
    depth = 0
    for i, tok in enumerate(tokens):
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0 and tok.upper() == "WITHIN":
            # Expect: WITHIN <dist> OF
            if i + 2 < len(tokens) and tokens[i + 2].upper() == "OF":
                dist = float(tokens[i + 1])
                left = tokens[:i]
                right = tokens[i + 3:]
                return left, dist, right
    return None


def select_atom_mask(atoms: Atoms, selection: str) -> np.ndarray:
    """Return a boolean mask over ``atoms`` for the given selection string."""
    tokens = _tokenize(selection)
    if not tokens:
        return np.ones(len(atoms), dtype=bool)

    within = _split_within(tokens)
    if within is not None:
        left_tokens, dist, right_tokens = within
        left_mask = (
            _Parser(left_tokens, atoms).parse()
            if left_tokens
            else np.ones(len(atoms), dtype=bool)
        )
        right_mask = (
            _Parser(right_tokens, atoms).parse()
            if right_tokens
            else np.ones(len(atoms), dtype=bool)
        )
        return _within(atoms, left_mask, dist, right_mask)

    return _Parser(tokens, atoms).parse()


def _within(
    atoms: Atoms, left_mask: np.ndarray, dist: float, right_mask: np.ndarray
) -> np.ndarray:
    """Atoms in ``left_mask`` within ``dist`` Å of any atom in ``right_mask``."""
    result = np.zeros(len(atoms), dtype=bool)
    if not right_mask.any() or not left_mask.any():
        return result
    right_xyz = atoms.xyz[right_mask]
    left_idx = np.flatnonzero(left_mask)
    left_xyz = atoms.xyz[left_idx]
    right_tree = cKDTree(right_xyz)
    # Query each left atom; keep those with at least one right neighbor.
    near = right_tree.query_ball_point(left_xyz, r=dist, return_length=True)
    result[left_idx[near > 0]] = True
    return result


def select_residue_mask(atoms: Atoms, selection: str) -> np.ndarray:
    """Return a per-position boolean mask (length ``num_positions``).

    Implements MSL's by-residue-CA rule: a position is selected iff its ``CA``
    atom is in the atom selection.
    """
    atom_mask = select_atom_mask(atoms, selection)
    num_pos = atoms.num_positions
    pos_mask = np.zeros(num_pos, dtype=bool)
    ca_atoms = atom_mask & (atoms.name == "CA")
    selected_positions = np.unique(atoms.position_index[ca_atoms])
    pos_mask[selected_positions] = True
    return pos_mask
