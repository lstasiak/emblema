r"""Put files in the artifact store as one transient archive, for a notebook cell or back from one.

A run on a paid card reads inputs no publication holds — the hourly means a script derived, the
cells it answers — and hands back what it wrote. Each goes as one compressed tar, kept for a
week: built the same way every time (names sorted, times, owners and modes fixed), so the same
files give the same digest. ``fetch_published.py archive`` unpacks it. Prints the key and its
checksum, one line.

    uv run --env-file .env.r2 scripts/put_archive.py --root DIR PATH [PATH ...] \\
        [--exclude GLOB ...]

Each path is a file or directory relative to ``--root``, kept under that name; a file whose path
matches an ``--exclude`` pattern stays out.
"""

import argparse
import fnmatch
import gzip
import io
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def files_of(root: Path, paths: Sequence[str], exclude: Sequence[str] = ()) -> list[str]:
    """Every file under the paths, relative to ``root``, sorted, none an exclusion matches.

    Raises:
        ValueError: If a path is missing, leaves the root, or nothing is left to archive.
    """
    found: set[str] = set()
    for name in paths:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"{name} lies outside {root}")
        if not path.exists():
            raise ValueError(f"{name} is not under {root}")
        found |= {
            str(p.relative_to(root.resolve()))
            for p in ([path] if path.is_file() else path.rglob("*"))
            if p.is_file()
        }
    kept = sorted(f for f in found if not any(fnmatch.fnmatch(f, g) for g in exclude))
    if not kept:
        raise ValueError("nothing to archive")
    return kept


def archived(root: Path, names: Sequence[str]) -> bytes:
    """The files as one gzip-compressed tar, the same bytes for the same files."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as packed,
        tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name in names:
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the store
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.shared.kernel.retention import Retention

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--exclude", nargs="*", default=[])
    arguments = parser.parse_args(argv)
    names = files_of(arguments.root, arguments.paths, arguments.exclude)
    data = archived(arguments.root, names)
    ref = configured_store(Settings()).put(data, Retention.TRANSIENT)
    print(f"{len(names)} files, {len(data) / 2**20:.1f} MB", file=sys.stderr)
    print(ref.key, ref.checksum)


if __name__ == "__main__":  # pragma: no cover
    main()
