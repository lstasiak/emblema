"""Fetch the raw corpora the data spike measures, from their sources of record.

Each archive is downloaded once into ``data/raw/<corpus>/``, hashed with SHA-256, checked against
the publisher's checksum where one exists (MD5 on Zenodo) and unpacked; a rerun skips finished work
and resumes interrupted downloads. A file failing its checksum is set aside as ``*.bad``, never
unpacked, and fetched again; any other failure is reported at the end without stopping the run.

    uv run scripts/fetch_corpora.py                # every corpus
    uv run scripts/fetch_corpora.py cmapss skab    # a selection
    uv run scripts/fetch_corpora.py --recheck      # hash the files again, trusting nothing
    uv run scripts/fetch_corpora.py --list

No mirrors: the source of record carries the licence and version; GitHub sources are commit-pinned.
"""

import argparse
import hashlib
import json
import struct
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK = 8 * 1024 * 1024
NESTED_WRAPPER_LIMIT = 3
RANGE_NOT_SATISFIABLE = 416
DIGEST_CACHE = ".digests.json"
# Compression method 9, Deflate64 (PKWARE "enhanced deflate"): zlib has no decompressor for it, so
# zipfile raises NotImplementedError on such a member. The ESA mission archives use it for one
# metadata CSV each, everything else in them is stored or deflated.
DEFLATE64 = 9
# Fixed part of a local file header, before the variable-length name and extra field.
LOCAL_HEADER = 30

# Plain dataclasses, not pydantic models: these are literals in code, nothing is parsed or
# validated at runtime.


@dataclass(frozen=True)
class Archive:
    """One downloadable file of a corpus.

    Attributes:
        url: Direct download URL at the source of record.
        filename: Local name under ``data/raw/<corpus>/``.
        md5: Publisher's checksum, when one is published; verified after download.
    """

    url: str
    filename: str
    md5: str | None = None


@dataclass(frozen=True)
class Corpus:
    """A corpus and the archives that make it up.

    Attributes:
        key: Directory name under ``data/raw/`` and the command-line selector.
        title: Human-readable name with the publisher.
        archives: Files to download, in order.
    """

    key: str
    title: str
    archives: tuple[Archive, ...]


# master of each repository on 2026-09-11; the archive URL below is immutable for a commit.
SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
OMNIANOMALY_COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
PHYSIONET = "https://physionet.org/files/challenge-2012/1.0.0"
ZENODO_ESA_AD_V2 = "https://zenodo.org/api/records/15237121/files"

CORPORA: tuple[Corpus, ...] = (
    Corpus(
        key="cmapss",
        title="C-MAPSS turbofan engine degradation simulation (NASA PCoE)",
        archives=(
            Archive(
                url=(
                    "https://phm-datasets.s3.amazonaws.com/NASA/"
                    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
                ),
                filename="turbofan-engine-degradation-simulation.zip",
            ),
        ),
    ),
    Corpus(
        key="skab",
        title="Skoltech Anomaly Benchmark (waico/SKAB)",
        archives=(
            Archive(
                url=f"https://github.com/waico/SKAB/archive/{SKAB_COMMIT}.zip",
                filename=f"SKAB-{SKAB_COMMIT[:12]}.zip",
            ),
        ),
    ),
    Corpus(
        key="smd",
        title="Server Machine Dataset (NetManAIOps/OmniAnomaly)",
        archives=(
            Archive(
                url=f"https://github.com/NetManAIOps/OmniAnomaly/archive/{OMNIANOMALY_COMMIT}.zip",
                filename=f"OmniAnomaly-{OMNIANOMALY_COMMIT[:12]}.zip",
            ),
        ),
    ),
    Corpus(
        key="physionet2012",
        title="PhysioNet/CinC Challenge 2012 (physionet.org)",
        archives=(
            Archive(url=f"{PHYSIONET}/set-a.zip", filename="set-a.zip"),
            Archive(url=f"{PHYSIONET}/set-b.zip", filename="set-b.zip"),
            Archive(url=f"{PHYSIONET}/set-c.tar.gz", filename="set-c.tar.gz"),
            Archive(url=f"{PHYSIONET}/Outcomes-a.txt", filename="Outcomes-a.txt"),
            Archive(url=f"{PHYSIONET}/Outcomes-b.txt", filename="Outcomes-b.txt"),
            Archive(url=f"{PHYSIONET}/Outcomes-c.txt", filename="Outcomes-c.txt"),
        ),
    ),
    Corpus(
        key="esa_ad",
        title="ESA Anomaly Dataset v2 (Zenodo 10.5281/zenodo.15237121)",
        archives=(
            Archive(
                url=f"{ZENODO_ESA_AD_V2}/ESA-Mission1.zip/content",
                filename="ESA-Mission1.zip",
                md5="9770ad12ed730238f37c42d5c27ab436",
            ),
            Archive(
                url=f"{ZENODO_ESA_AD_V2}/ESA-Mission2.zip/content",
                filename="ESA-Mission2.zip",
                md5="bfc72012691427d9327eb41f726ce45e",
            ),
            Archive(
                url=f"{ZENODO_ESA_AD_V2}/ESA-Mission3.zip/content",
                filename="ESA-Mission3.zip",
                md5="d63943f09c81378acd9fc5e565ecc66e",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class Fetched:
    """One archive after a run, whether or not it arrived.

    Attributes:
        corpus: Corpus key.
        filename: Local name under ``data/raw/<corpus>/``.
        size: Bytes on disk, 0 for an archive that never arrived.
        sha256: Digest of the file, empty for an archive that never arrived.
        md5_status: Verdict against the publisher's checksum.
        error: Why the archive is missing, when it is; None once it is on disk.
    """

    corpus: str
    filename: str
    size: int
    sha256: str
    md5_status: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.md5_status.startswith("MISMATCH")

    @property
    def problem(self) -> str:
        return self.error or self.md5_status


class Digests(TypedDict):
    size: int
    sha256: str
    md5: str


def note(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def download(url: str, target: Path) -> None:
    """Download ``url`` to ``target``, resuming a partial file if one is left over."""
    if target.exists():
        note(f"{target.name}: already downloaded")
        return
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "emblema-fetch-corpora"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request) as response:
            resumed = response.status == 206
            total = response.headers.get("Content-Length")
            expected = ((offset if resumed else 0) + int(total)) if total else None
            with partial.open("ab" if resumed else "wb") as sink:
                done = offset if resumed else 0
                for chunk in iter(lambda: response.read(CHUNK), b""):
                    sink.write(chunk)
                    done += len(chunk)
                    progress(target.name, done, expected)
    except urllib.error.HTTPError as error:
        # The partial file already holds every byte: a run that died between the last chunk and
        # the rename leaves exactly this state behind.
        if not (offset and error.code == RANGE_NOT_SATISFIABLE):
            raise
    sys.stderr.write("\n")
    partial.rename(target)


def progress(name: str, done: int, expected: int | None) -> None:
    if expected:
        sys.stderr.write(f"\r{name}: {done / 2**20:,.0f} / {expected / 2**20:,.0f} MiB")
    else:
        sys.stderr.write(f"\r{name}: {done / 2**20:,.0f} MiB")


def digests(path: Path) -> tuple[str, str]:
    """SHA-256 and MD5 of a file in one pass; both are needed and the files are large."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def remembered(directory: Path) -> dict[str, Digests]:
    """Checksums an earlier run wrote down, by filename; an unreadable note is no note at all."""
    path = directory / DIGEST_CACHE
    if not path.exists():
        return {}
    try:
        cache: dict[str, Digests] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return cache


def remember(directory: Path, cache: dict[str, Digests]) -> None:
    path = directory / DIGEST_CACHE
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def forget(path: Path) -> None:
    """Drop a remembered checksum, so a replacement file is hashed again."""
    cache = remembered(path.parent)
    if cache.pop(path.name, None) is not None:
        remember(path.parent, cache)


def checksums(path: Path, *, recheck: bool = False) -> tuple[str, str]:
    """SHA-256 and MD5 of a file, remembered so a re-run does not hash 11.6 GB a second time.

    The size on disk guards the memory: a file that grew, shrank or was replaced since it was
    hashed is hashed again, and ``--recheck`` distrusts the note entirely.
    """
    size = path.stat().st_size
    cache = remembered(path.parent)
    entry = cache.get(path.name)
    if entry is not None and entry["size"] == size and not recheck:
        note(f"{path.name}: checksums remembered")
        return entry["sha256"], entry["md5"]
    sha256, md5 = digests(path)
    cache[path.name] = Digests(size=size, sha256=sha256, md5=md5)
    remember(path.parent, cache)
    return sha256, md5


def checksum_status(published: str | None, actual: str) -> str:
    if published is None:
        return "none published"
    return "ok" if actual == published else f"MISMATCH (got {actual})"


def unpack(archive: Path, into: Path) -> None:
    """Extract an archive once, then the zips inside a thin wrapper (NASA ships a zip in a zip).

    A wrapper holds a few files; an archive with hundreds of inner zips, such as a satellite
    mission with one zip per channel, is data and stays compressed. Finder metadata that macOS
    puts into zips (``__MACOSX``, ``._`` resource forks) is neither.
    """
    marker = into / f".unpacked-{archive.name}"
    if marker.exists():
        note(f"{archive.name}: already unpacked")
        return
    extracted = [path for path in extract(archive, into) if "__MACOSX" not in path.parts]
    nested = [
        path
        for path in extracted
        if path.suffix == ".zip" and not path.name.startswith("._") and zipfile.is_zipfile(path)
    ]
    if nested and len(extracted) <= NESTED_WRAPPER_LIMIT:
        for inner in nested:
            list(extract(inner, inner.with_suffix("")))
    marker.touch()


def extract(archive: Path, into: Path) -> Iterator[Path]:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        yield from extract_zip(archive, into)
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(into, filter="data")
            for member in bundle.getmembers():
                if member.isfile():
                    yield into / member.name


def extract_zip(archive: Path, into: Path) -> Iterator[Path]:
    """Extract a zip, decompressing any Deflate64 member ourselves.

    ``zipfile`` refuses method 9 outright, and refuses it in the middle of ``extractall``, which
    leaves the archive half unpacked. Those members are handed to ``inflate64`` one at a time
    instead; they are metadata files of a few kilobytes, so reading one whole is cheap.
    """
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        bundle.extractall(into, members=[m for m in members if m.compress_type != DEFLATE64])
    for member in members:
        if member.compress_type == DEFLATE64:
            write_deflate64(archive, member, into)
    for member in members:
        if not member.is_dir():
            yield member_path(into, member)


def member_path(into: Path, member: zipfile.ZipInfo) -> Path:
    """Where ``extractall`` puts a member: its name, without any leading or upward path."""
    parts = [
        part
        for part in Path(member.filename.replace("\\", "/")).parts
        if part not in ("/", ".", "..")
    ]
    return into.joinpath(*parts)


def write_deflate64(archive: Path, member: zipfile.ZipInfo, into: Path) -> Path:
    # inflate64 is needed by this script alone, for one metadata file per ESA mission: imported
    # here so that fetching every other corpus works without it.
    import inflate64

    with archive.open("rb") as source:
        source.seek(member.header_offset)
        header = source.read(LOCAL_HEADER)
        name_length, extra_length = struct.unpack("<HH", header[26:LOCAL_HEADER])
        source.seek(member.header_offset + LOCAL_HEADER + name_length + extra_length)
        payload = inflate64.Inflater().inflate(source.read(member.compress_size))
    if len(payload) != member.file_size or zlib.crc32(payload) != member.CRC:
        raise ValueError(f"{archive.name}: {member.filename} is corrupt after decompression")
    target = member_path(into, member)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def fetch(corpus: Corpus, *, recheck: bool = False) -> Iterator[Fetched]:
    directory = DATA_ROOT / corpus.key
    directory.mkdir(parents=True, exist_ok=True)
    for archive in corpus.archives:
        try:
            yield fetch_archive(corpus, archive, directory, recheck=recheck)
        except Exception as error:
            # One archive is one file of one corpus. The rest of the run is still worth doing, and
            # the checksums of what did arrive are still worth reporting.
            note(f"{archive.filename}: FAILED, {error}")
            yield Fetched(corpus.key, archive.filename, 0, "", "not verified", f"{error}")


def fetch_archive(corpus: Corpus, archive: Archive, directory: Path, *, recheck: bool) -> Fetched:
    target = directory / archive.filename
    download(archive.url, target)
    sha256, md5 = checksums(target, recheck=recheck)
    row = Fetched(
        corpus.key,
        archive.filename,
        target.stat().st_size,
        sha256,
        checksum_status(archive.md5, md5),
    )
    if not row.ok:
        forget(target)
        target.replace(target.with_suffix(target.suffix + ".bad"))
    elif archive.filename.endswith((".zip", ".tar.gz")):
        unpack(target, directory)
    return row


def report(rows: Iterable[Fetched]) -> str:
    lines = [
        "| Corpus | File | Bytes | SHA-256 | Publisher MD5 |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row.corpus} | {row.filename} | {row.size:,} | `{row.sha256}` | {row.md5_status} |"
        for row in rows
        if row.error is None
    )
    return "\n".join(lines)


def problems(rows: Iterable[Fetched]) -> str:
    lines = ["Not done, run again (a file that failed its published checksum is set aside):"]
    lines.extend(f"- {row.corpus}/{row.filename}: {row.problem}" for row in rows if not row.ok)
    return "\n".join(lines)


def select(keys: list[str]) -> tuple[Corpus, ...]:
    if not keys:
        return CORPORA
    known = {corpus.key: corpus for corpus in CORPORA}
    unknown = sorted(set(keys) - known.keys())
    if unknown:
        raise SystemExit(f"unknown corpus: {', '.join(unknown)}; known: {', '.join(known)}")
    return tuple(known[key] for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpora", nargs="*", help="corpus keys; default: all")
    parser.add_argument("--list", action="store_true", help="list corpora and exit")
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="hash every file again instead of trusting the remembered checksums",
    )
    args = parser.parse_args()
    if args.list:
        for corpus in CORPORA:
            print(f"{corpus.key:14} {corpus.title}")
            for archive in corpus.archives:
                print(f"{'':14} {archive.url}")
        return
    rows = [row for corpus in select(args.corpora) for row in fetch(corpus, recheck=args.recheck)]
    print(f"\nData root: {DATA_ROOT}\n")
    print(report(rows))
    failed = [row for row in rows if not row.ok]
    if failed:
        print(f"\n{problems(rows)}")
        raise SystemExit(f"{len(failed)} of {len(rows)} archives need another run")


if __name__ == "__main__":
    main()
