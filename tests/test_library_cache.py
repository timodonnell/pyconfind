"""DEFAULT_ROTAMER_LIBRARY: lazy, process-wide cache for the auto-downloaded library.

Parsing EBL.out costs ~5 s on a typical machine; without this cache every
``analyze("foo.pdb")`` call would re-parse it. Power users who want their own
library should build a :class:`RotamerLibrary` once via :func:`load_library`
and pass it explicitly to every ``analyze`` call.
"""

from __future__ import annotations

from pathlib import Path

import pyconfind.api as api
from pyconfind import analyze, load_library

MINI_ROTLIB = Path(__file__).resolve().parent / "data" / "mini_rotlib"
PDB = Path(__file__).resolve().parent / "data" / "structures" / "1UBQ.pdb"


def test_cache_is_lazy_and_reused(monkeypatch) -> None:
    """First call with no library populates the cache; subsequent calls reuse it."""
    api.DEFAULT_ROTAMER_LIBRARY = None
    # Avoid hitting the network / the user's real cache: redirect the auto-loader
    # to the bundled mini library.
    monkeypatch.setattr(
        api, "cached_rotamer_library", lambda: MINI_ROTLIB,
    )

    n_loads = 0
    real_load = api.load_library

    def counting_load(p):
        nonlocal n_loads
        n_loads += 1
        return real_load(p)

    monkeypatch.setattr(api, "load_library", counting_load)

    a1 = analyze(PDB)
    a2 = analyze(PDB)
    assert n_loads == 1, f"expected one parse, got {n_loads}"
    assert a1.library is a2.library is api.DEFAULT_ROTAMER_LIBRARY


def test_explicit_library_bypasses_cache(monkeypatch) -> None:
    """Passing rotamer_library= must use that exact object, untouched by the cache."""
    api.DEFAULT_ROTAMER_LIBRARY = None  # ensure the default isn't accidentally returned
    user_lib = load_library(MINI_ROTLIB)
    a = analyze(PDB, rotamer_library=user_lib)
    assert a.library is user_lib
    assert api.DEFAULT_ROTAMER_LIBRARY is None  # custom library doesn't seed the default


def test_resetting_cache_triggers_reload(monkeypatch) -> None:
    api.DEFAULT_ROTAMER_LIBRARY = None
    monkeypatch.setattr(api, "cached_rotamer_library", lambda: MINI_ROTLIB)
    a1 = analyze(PDB)
    api.DEFAULT_ROTAMER_LIBRARY = None
    a2 = analyze(PDB)
    assert a1.library is not a2.library
