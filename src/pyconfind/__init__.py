"""pyconfind — protein side-chain contact-degree analysis."""

from .api import Analysis, analyze
from .output import OutputOptions, format_confind_text, format_json
from .pdb import read_pdb, read_structure
from .rotlib import RotamerLibrary, load_library

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "OutputOptions",
    "RotamerLibrary",
    "analyze",
    "format_confind_text",
    "format_json",
    "load_library",
    "read_pdb",
    "read_structure",
]
