"""Helpers for obtaining the rotamer library.

The Dunbrack 2010 rotamer library that confind uses (~6 MB packaged) is
distributed as an asset on the pyconfind GitHub releases rather than bundled
in the wheel. :func:`download_rotamer_library` fetches and extracts it.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

# The library asset is static, so it is hosted once on the v0.1.0 release and
# reused by every package version (no need to re-upload it on each release).
_ROTLIB_RELEASE = "v0.1.0"

#: URL of the packaged rotamer library (``rotlibs/EBL.out`` + ``BEBL.out``)
#: attached to the pyconfind GitHub release.
ROTAMER_LIBRARY_URL = (
    f"https://github.com/timodonnell/pyconfind/releases/download/"
    f"{_ROTLIB_RELEASE}/rotlibs.tar.gz"
)


def download_rotamer_library(
    dest: str | Path = "rotlibs",
    *,
    url: str = ROTAMER_LIBRARY_URL,
    force: bool = False,
) -> Path:
    """Download and extract the rotamer library, returning its directory.

    Parameters
    ----------
    dest
        Directory to extract the library into (created if needed). If it
        already contains ``EBL.out`` and ``BEBL.out`` the download is skipped
        unless ``force`` is set.
    url
        Override the download URL (e.g. to pin a different release).
    force
        Re-download even if the destination already looks populated.

    Returns
    -------
    Path to a directory containing ``EBL.out`` and ``BEBL.out``, suitable to
    pass as ``rotamer_library=`` to :func:`pyconfind.analyze`.
    """
    dest = Path(dest)
    if not force and (dest / "EBL.out").exists() and (dest / "BEBL.out").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"pyconfind: downloading rotamer library to {dest} ...", file=sys.stderr)
    tmp_tar, _ = urllib.request.urlretrieve(url)  # noqa: S310 (trusted release URL)
    try:
        with tarfile.open(tmp_tar) as tar:
            # The archive stores the files under ``rotlibs/``; extract those
            # two files directly into ``dest``.
            members = [
                m for m in tar.getmembers()
                if Path(m.name).name in ("EBL.out", "BEBL.out")
            ]
            if not members:
                raise RuntimeError(f"no EBL.out/BEBL.out found in {url}")
            dest.mkdir(parents=True, exist_ok=True)
            for m in members:
                fobj = tar.extractfile(m)
                if fobj is None:
                    continue
                (dest / Path(m.name).name).write_bytes(fobj.read())
    finally:
        Path(tmp_tar).unlink(missing_ok=True)
    return dest


def cached_rotamer_library(*, url: str = ROTAMER_LIBRARY_URL) -> Path:
    """Return the rotamer library from a per-user cache, downloading on first use.

    The library is stored under the platform cache directory
    (``platformdirs.user_cache_dir("pyconfind")/rotlibs``) and reused across
    runs and projects. This is what :func:`pyconfind.analyze` and the CLI use
    when no ``rotamer_library`` / ``--rLib`` is given.
    """
    import platformdirs

    dest = Path(platformdirs.user_cache_dir("pyconfind")) / "rotlibs"
    return download_rotamer_library(dest=dest, url=url)
