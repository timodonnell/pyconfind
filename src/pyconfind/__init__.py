"""pyconfind — protein side-chain contact-degree analysis."""

from .api import Analysis, analyze
from .data import ROTAMER_LIBRARY_URL, download_rotamer_library
from .output import OutputOptions, format_confind_text, format_json
from .pdb import read_pdb, read_structure
from .rotlib import RotamerLibrary, load_library

__version__ = "0.1.0"

__all__ = [
    "ROTAMER_LIBRARY_URL",
    "Analysis",
    "OutputOptions",
    "RotamerLibrary",
    "analyze",
    "download_rotamer_library",
    "format_confind_text",
    "format_json",
    "load_library",
    "read_pdb",
    "read_structure",
]
