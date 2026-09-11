"""Fetch the raw corpora the data spike measures, from their sources of record.

Each archive is downloaded once into ``data/raw/<corpus>/``, hashed with SHA-256 for the
verification note, checked against the publisher's checksum where one is published (Zenodo gives
MD5), and unpacked next to it. Interrupted downloads resume where they stopped, which matters for
the 11.6 GB of satellite telemetry. Re-running is idempotent. A file that fails its published
checksum is set aside as ``*.bad``, never unpacked, and fetched again on the next run.

    uv run scripts/fetch_corpora.py                # every corpus
    uv run scripts/fetch_corpora.py cmapss skab    # a selection
    uv run scripts/fetch_corpora.py --list

Mirrors (Kaggle re-uploads, forks) are deliberately absent: the source of record carries the licence
and the version. GitHub-hosted corpora are pinned to a commit for the same reason.
"""

import argparse
import hashlib
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK = 8 * 1024 * 1024
NESTED_WRAPPER_LIMIT = 3
RANGE_NOT_SATISFIABLE = 416

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
    corpus: str
    filename: str
    size: int
    sha256: str
    md5_status: str

    @property
    def ok(self) -> bool:
        return not self.md5_status.startswith("MISMATCH")


def download(url: str, target: Path) -> None:
    """Download ``url`` to ``target``, resuming a partial file if one is left over."""
    if target.exists():
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
            expected = (offset if resumed else 0) + int(total) if total else None
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
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
            for name in bundle.namelist():
                if not name.endswith("/"):
                    yield into / name
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(into, filter="data")
            for member in bundle.getmembers():
                if member.isfile():
                    yield into / member.name


def fetch(corpus: Corpus) -> Iterator[Fetched]:
    directory = DATA_ROOT / corpus.key
    directory.mkdir(parents=True, exist_ok=True)
    for archive in corpus.archives:
        target = directory / archive.filename
        download(archive.url, target)
        sha256, md5 = digests(target)
        row = Fetched(
            corpus.key,
            archive.filename,
            target.stat().st_size,
            sha256,
            checksum_status(archive.md5, md5),
        )
        if not row.ok:
            target.replace(target.with_suffix(target.suffix + ".bad"))
        elif archive.filename.endswith((".zip", ".tar.gz")):
            unpack(target, directory)
        yield row


def report(rows: Iterable[Fetched]) -> str:
    lines = [
        "| Corpus | File | Bytes | SHA-256 | Publisher MD5 |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row.corpus} | {row.filename} | {row.size:,} | `{row.sha256}` | {row.md5_status} |"
        for row in rows
    )
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
    args = parser.parse_args()
    if args.list:
        for corpus in CORPORA:
            print(f"{corpus.key:14} {corpus.title}")
            for archive in corpus.archives:
                print(f"{'':14} {archive.url}")
        return
    rows = [row for corpus in select(args.corpora) for row in fetch(corpus)]
    print(f"\nData root: {DATA_ROOT}\n")
    print(report(rows))
    failed = [row.filename for row in rows if not row.ok]
    if failed:
        raise SystemExit(f"checksum mismatch, set aside as .bad: {', '.join(failed)}")


if __name__ == "__main__":
    main()
