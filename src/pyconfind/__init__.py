"""pyconfind — protein side-chain contact-degree analysis."""

from .api import Analysis, analyze
from .data import (
    ROTAMER_LIBRARY_URL,
    cached_rotamer_library,
    download_rotamer_library,
)
from .output import (
    OutputOptions,
    ParsedConfind,
    format_confind_text,
    format_json,
    parse_confind_text,
)
from .pdb import read_pdb, read_structure
from .rotlib import RotamerLibrary, load_library

__version__ = "0.5.0"

__all__ = [
    "ROTAMER_LIBRARY_URL",
    "Analysis",
    "OutputOptions",
    "ParsedConfind",
    "RotamerLibrary",
    "analyze",
    "cached_rotamer_library",
    "download_rotamer_library",
    "format_confind_text",
    "format_json",
    "load_library",
    "parse_confind_text",
    "read_pdb",
    "read_structure",
]
