"""Output formatters for pyconfind.

Two formats are provided:

* :func:`format_confind_text` — the original confind line-oriented format,
  byte-compatible with the C++ reference where possible.

* :func:`format_json` — a structured representation suitable for downstream
  pipelines, notebooks, and modern tools.

The text format consists of, in order:

* ``contact <pos_i> <pos_j> <degree> <resname_i> <resname_j>`` per pair
* ``sumcond <pos> <degree> [<phi> <psi>] [<omega>] <resname> [<pdb>]``
* ``percont <pos_i> <pos_j> -1.000000 <resname_i> <resname_j>`` per permanent contact
* ``crwdnes <pos> <fraction> [<phi> <psi>] [<omega>] <resname> [<pdb>]``
* ``freedom <pos> <value> [<phi> <psi>] [<omega>] <resname> [<pdb>]``
* ``SEQUENCE: <res1> <res2> ...``

Position IDs are ``<chain>,<resnum><icode>``; undefined phi/psi/omega is 999.0.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .build import PositionRotamers
from .contacts import ContactReport


@dataclass(frozen=True)
class ParsedConfind:
    """Parsed confind text output (e.g. a C++ ``.cont`` file).

    Lets you read a reference contact map back in — the inverse of
    :func:`format_confind_text` — for comparison or plotting.
    """

    #: ``{(pos_i_id, pos_j_id): degree}`` for ``contact`` rows.
    contacts: dict[tuple[str, str], float] = field(default_factory=dict)
    #: ``{pos_id: value}`` for each per-position row type.
    sumcond: dict[str, float] = field(default_factory=dict)
    crwdnes: dict[str, float] = field(default_factory=dict)
    freedom: dict[str, float] = field(default_factory=dict)
    #: ``[(pos_i_id, pos_j_id)]`` for ``percont`` rows.
    percont: list[tuple[str, str]] = field(default_factory=list)
    #: Position ids in output order (from the per-position rows).
    order: list[str] = field(default_factory=list)


def parse_confind_text(text: str) -> ParsedConfind:
    """Parse confind ``.cont`` text (C++ or pyconfind output) into a structure.

    Tolerant of the optional phi/psi/omega/filename columns: per-position rows
    are ``<tag> <pos> <value> [..extra..] <resname> [..]`` so the value is
    always field index 2.
    """
    out = ParsedConfind()
    seen: set[str] = set()
    for line in text.splitlines():
        parts = line.split("\t")
        tag = parts[0]
        if tag == "contact" and len(parts) >= 4:
            out.contacts[(parts[1], parts[2])] = float(parts[3])
        elif tag == "percont" and len(parts) >= 3:
            out.percont.append((parts[1], parts[2]))
        elif tag in ("sumcond", "crwdnes", "freedom") and len(parts) >= 3:
            pos = parts[1]
            val = float(parts[2]) if parts[2] not in ("-nan", "nan") else math.nan
            getattr(out, tag)[pos] = val
            if tag == "sumcond" and pos not in seen:
                seen.add(pos)
                out.order.append(pos)
    return out


@dataclass(frozen=True)
class OutputOptions:
    """Toggles mirroring the C++ ``--pp``, ``--omg``, ``--pf`` flags."""

    include_phi_psi: bool = False
    include_omega: bool = False
    pdb_filename: str | None = None  # echoed back when --pf is set


# Immutable module-level singleton used as the default for formatter args
# (a frozen dataclass with all-default fields is safe to share).
_DEFAULT_OPTIONS = OutputOptions()


def _position_id(pr: PositionRotamers) -> str:
    pos = pr.position
    return f"{pos.chain},{pos.resnum}{pos.icode}"


def _format_float(value: float) -> str:
    """Match ``std::setprecision(6) << std::fixed << value`` from MSL.

    NaN renders as ``-nan`` to mirror glibc's printf output that the C++
    binary emits.
    """
    if math.isnan(value):
        return "-nan"
    return f"{value:.6f}"


def _format_pp(value: float | None) -> str:
    """Render an optional phi/psi/omega angle.

    ``None`` (undefined dihedral) becomes ``999.000000``; any other value is
    formatted with 6 fixed decimals.
    """
    v = 999.0 if value is None else value
    return _format_float(v)


def format_confind_text(
    positions: list[PositionRotamers],
    report: ContactReport,
    options: OutputOptions = _DEFAULT_OPTIONS,
) -> str:
    """Render the confind output as the original text format.

    Matches the row ordering and field layout of the C++ ``confind`` binary,
    so downstream tools that parse the original output can use pyconfind as
    a drop-in replacement.
    """
    lines: list[str] = []
    # 1) contact rows.
    for c in report.contacts:
        pi = positions[c.pos_i]
        pj = positions[c.pos_j]
        lines.append(
            "\t".join(
                (
                    "contact",
                    _position_id(pi),
                    _position_id(pj),
                    _format_float(c.degree),
                    pi.position.resname,
                    pj.position.resname,
                )
            )
        )

    # When --sel is in effect, only in-focus positions are emitted.
    focus_idx = [i for i, pr in enumerate(positions) if pr.in_focus]

    # 2) sumcond rows.
    for i in focus_idx:
        lines.append(
            _per_position_row("sumcond", positions[i], report.sum_contact_degree[i], options)
        )

    # 3) percont rows. The C++ writes one row per unordered (i, j) permanent
    # contact in the same direction it was discovered (i -> contacts of i).
    for i in focus_idx:
        pr = positions[i]
        for j in sorted(pr.permanent_contacts):
            other = positions[j]
            lines.append(
                "\t".join(
                    (
                        "percont",
                        _position_id(pr),
                        _position_id(other),
                        "-1.000000",
                        pr.position.resname,
                        other.position.resname,
                    )
                )
            )

    # 4) crwdnes rows.
    for i in focus_idx:
        lines.append(_per_position_row("crwdnes", positions[i], report.crwdnes[i], options))

    # 5) freedom rows.
    for i in focus_idx:
        lines.append(_per_position_row("freedom", positions[i], report.freedom[i], options))

    # 6) SEQUENCE: trailing row.
    seq = " ".join(positions[i].position.resname for i in focus_idx)
    lines.append("SEQUENCE: " + seq)
    return "\n".join(lines) + "\n"


def _per_position_row(
    tag: str, pr: PositionRotamers, value: float, options: OutputOptions
) -> str:
    pos = pr.position
    fields: list[str] = [tag, _position_id(pr), _format_float(value)]
    if options.include_phi_psi:
        fields.append(_format_pp(pos.phi))
        fields.append(_format_pp(pos.psi))
    if options.include_omega:
        fields.append(_format_pp(pos.omega))
    fields.append(pos.resname)
    if options.pdb_filename is not None:
        fields.append(options.pdb_filename)
    return "\t".join(fields)


def format_json(
    positions: list[PositionRotamers],
    report: ContactReport,
    options: OutputOptions = _DEFAULT_OPTIONS,
    *,
    indent: int | None = 2,
) -> str:
    """Render as JSON. The structured format is the modern default.

    The JSON shape is:

    .. code-block:: json

        {
          "positions": [
            {"chain": "A", "resnum": 1, "icode": "", "resname": "ALA",
             "phi": null, "psi": 180.0, "omega": null,
             "sumcond": 0.009889, "crwdnes": 0.042328, "freedom": 0.955030,
             "permanent_contacts": []},
            ...
          ],
          "contacts": [
            {"i": {"chain": "A", "resnum": 1, "icode": "", "resname": "ALA"},
             "j": {"chain": "A", "resnum": 3, "icode": "", "resname": "ALA"},
             "degree": 0.009889},
            ...
          ],
          "sequence": ["ALA", "ILE", "ALA"]
        }
    """
    positions_json = []
    for i, pr in enumerate(positions):
        pos = pr.position
        positions_json.append(
            {
                "chain": pos.chain,
                "resnum": pos.resnum,
                "icode": pos.icode,
                "resname": pos.resname,
                "phi": pos.phi,
                "psi": pos.psi,
                "omega": pos.omega,
                "sumcond": _to_json_float(report.sum_contact_degree[i]),
                "crwdnes": _to_json_float(report.crwdnes[i]),
                "freedom": _to_json_float(report.freedom[i]),
                "permanent_contacts": sorted(
                    {
                        f"{positions[j].position.chain},"
                        f"{positions[j].position.resnum}"
                        f"{positions[j].position.icode}"
                        for j in pr.permanent_contacts
                    }
                ),
            }
        )
    contacts_json = []
    for c in report.contacts:
        pi = positions[c.pos_i].position
        pj = positions[c.pos_j].position
        contacts_json.append(
            {
                "i": {
                    "chain": pi.chain, "resnum": pi.resnum,
                    "icode": pi.icode, "resname": pi.resname,
                },
                "j": {
                    "chain": pj.chain, "resnum": pj.resnum,
                    "icode": pj.icode, "resname": pj.resname,
                },
                "degree": float(c.degree),
            }
        )
    payload: dict[str, object] = {
        "positions": positions_json,
        "contacts": contacts_json,
        "sequence": [pr.position.resname for pr in positions],
    }
    if options.pdb_filename is not None:
        payload["pdb_filename"] = options.pdb_filename
    return json.dumps(payload, indent=indent, allow_nan=True)


def _to_json_float(value: float) -> float | None:
    """NaN goes to ``null`` for JSON serialization."""
    f = float(value)
    return None if math.isnan(f) else f


def iter_contact_rows(positions: list[PositionRotamers], report: ContactReport) -> Iterable[str]:
    """Yield the ``contact`` rows only — useful when streaming large outputs."""
    for c in report.contacts:
        pi = positions[c.pos_i]
        pj = positions[c.pos_j]
        yield "\t".join(
            (
                "contact",
                _position_id(pi),
                _position_id(pj),
                _format_float(c.degree),
                pi.position.resname,
                pj.position.resname,
            )
        )
