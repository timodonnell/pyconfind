"""Command-line interface for pyconfind.

Aims for drop-in compatibility with the C++ ``confind`` flags so existing
pipelines work unchanged:

* ``--p``       input PDB (or use ``--pL`` for a list file)
* ``--o``       output file
* ``--rLib``    rotamer library directory (optional; auto-downloaded + cached if omitted)
* ``--pp``      include phi/psi in per-position rows
* ``--omg``     include omega in per-position rows
* ``--pf``      append the PDB filename to per-position rows
* ``--ren``     renumber residues per chain

pyconfind extensions:

* ``--json``         emit structured JSON instead of the legacy text format
* ``--native-only``  only substitute the native AA at each position
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .api import analyze
from .data import cached_rotamer_library
from .output import OutputOptions, format_confind_text, format_json
from .rotlib import load_library


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    name="pyconfind",
)
@click.option("--p", "pdb_file", type=click.Path(exists=True, path_type=Path), help="Input structure file (PDB or mmCIF; format auto-detected).")
@click.option("--pL", "pdb_list_file", type=click.Path(exists=True, path_type=Path), help="File listing PDB paths, one per line.")
@click.option("--o", "out_file", type=click.Path(path_type=Path), help="Output file path. Stdout if omitted.")
@click.option("--oL", "out_list_file", type=click.Path(path_type=Path), help="File listing output paths for batch mode.")
@click.option("--rLib", "rotlib_path", type=click.Path(exists=True, path_type=Path), default=None, help="Rotamer library directory (EBL.out + BEBL.out). If omitted, the Dunbrack 2010 library is downloaded once and cached per-user.")
@click.option("--pp", "include_pp", is_flag=True, help="Include phi/psi in per-position rows.")
@click.option("--omg", "include_omega", is_flag=True, help="Include omega in per-position rows.")
@click.option("--pf", "print_filename", is_flag=True, help="Append the PDB filename to per-position rows.")
@click.option("--psel", "pre_select", type=str, default=None, help="Pre-selection: keep only residues whose CA satisfies this selection.")
@click.option("--sel", "focus", type=str, default=None, help="Focus: compute/output only residues whose CA satisfies this selection (rest kept for clash detection).")
@click.option("--ren", "renumber", is_flag=True, help="Renumber residues per chain starting from 1.")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON instead of the legacy text format.")
@click.option("--native-only", "native_only", is_flag=True, help="At each position, only place rotamers of the native AA.")
@click.option("--dcut", "dcut", type=float, default=25.0, show_default=True, help="CA-CA distance cutoff for pair consideration.")
@click.option("--contact-dist", "contact_dist", type=float, default=3.0, show_default=True, help="Sidechain-sidechain contact distance cutoff.")
@click.option("--clash-dist", "clash_dist", type=float, default=2.0, show_default=True, help="Backbone-clash distance cutoff.")
@click.option("--backend", "backend", type=click.Choice(["auto", "numba", "python"]), default="auto", show_default=True, help="Contact-degree backend (numba = JIT/multithreaded, python = reference).")
@click.option("--assembly", "assembly", type=str, default="1", show_default=True, help="Biological assembly to analyze. Use 'au' for the asymmetric unit as-is.")
def main(
    pdb_file: Path | None,
    pdb_list_file: Path | None,
    out_file: Path | None,
    out_list_file: Path | None,
    rotlib_path: Path | None,
    include_pp: bool,
    include_omega: bool,
    print_filename: bool,
    pre_select: str | None,
    focus: str | None,
    renumber: bool,
    json_output: bool,
    native_only: bool,
    dcut: float,
    contact_dist: float,
    clash_dist: float,
    backend: str,
    assembly: str,
) -> None:
    """Analyze rotamer contacts in protein structures.

    Reproduces the original ``confind`` text output byte-for-byte; pass
    ``--json`` for a structured representation suitable for modern pipelines.
    """
    if pdb_file is None and pdb_list_file is None:
        raise click.UsageError("must specify either --p or --pL")
    if pdb_file is not None and pdb_list_file is not None:
        raise click.UsageError("--p and --pL are mutually exclusive")

    if pdb_file is not None:
        pdb_paths = [pdb_file]
    else:
        assert pdb_list_file is not None
        pdb_paths = [
            Path(line.strip())
            for line in pdb_list_file.read_text().splitlines()
            if line.strip()
        ]

    if out_list_file is not None:
        out_paths: list[Path | None] = [
            Path(line.strip())
            for line in out_list_file.read_text().splitlines()
            if line.strip()
        ]
    elif out_file is not None:
        if len(pdb_paths) > 1:
            base = out_file.with_suffix("")
            ext = out_file.suffix or ".cont"
            out_paths = [
                base.with_name(f"{base.name}.f{i+1}{ext}")
                for i in range(len(pdb_paths))
            ]
        else:
            out_paths = [out_file]
    else:
        out_paths = [None] * len(pdb_paths)

    # Pre-load the library once so we don't re-parse EBL.out per PDB. With no
    # --rLib, fall back to the per-user cached Dunbrack library (downloaded on
    # first use).
    library = load_library(rotlib_path if rotlib_path is not None else cached_rotamer_library())

    assembly_arg: str | None = None if assembly.lower() in ("au", "") else assembly
    for pdb_path, out_path in zip(pdb_paths, out_paths, strict=True):
        analysis = analyze(
            pdb_path,
            rotamer_library=library,
            pre_select=pre_select,
            focus=focus,
            renumber=renumber,
            native_only=native_only,
            dcut=dcut,
            contact_distance=contact_dist,
            clash_distance=clash_dist,
            backend=backend,
            assembly=assembly_arg,
        )
        options = OutputOptions(
            include_phi_psi=include_pp,
            include_omega=include_omega,
            pdb_filename=str(pdb_path) if print_filename else None,
        )
        if json_output:
            rendered = format_json(analysis.positions, analysis.report, options)
        else:
            rendered = format_confind_text(
                analysis.positions, analysis.report, options
            )

        if out_path is None:
            # Mirror the C++: when no --o is given, prepend the PDB filename
            # as a banner line, then the body.
            if not json_output:
                sys.stdout.write(f"{pdb_path}\n")
            sys.stdout.write(rendered)
            if not rendered.endswith("\n"):
                sys.stdout.write("\n")
        else:
            out_path.write_text(rendered)


if __name__ == "__main__":
    main()
