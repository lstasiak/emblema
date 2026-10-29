"""Fetch the raw corpora the data spike measures, from their sources of record.

Each archive is downloaded once into ``data/raw/<corpus>/``, hashed with SHA-256, checked against
the publisher's checksum where one exists (MD5 on Zenodo) and unpacked; a rerun skips finished work
and resumes interrupted downloads. A file failing its checksum is set aside as ``*.bad``, never
unpacked, and fetched again; any other failure is reported at the end without stopping the run.

    uv run scripts/fetch_corpora.py                # every corpus
    uv run scripts/fetch_corpora.py cmapss skab    # a selection
    uv run scripts/fetch_corpora.py --recheck      # hash the files again, trusting nothing
    uv run scripts/fetch_corpora.py --list
    uv run --env-file .env.r2 scripts/fetch_corpora.py --mirror           # copy to the store
    uv run scripts/fetch_corpora.py --from-store physionet2012             # read the copy

The source of record carries the licence and version; GitHub sources are commit-pinned. A corpus
under an open licence may also be mirrored in the artifact store, because its source can stall a
transfer for hours and a paid accelerator waits on it: ``--mirror`` copies the files there and
pins each copy's SHA-256 in ``raw_mirror.sha256``, and ``--from-store`` reads a pinned copy before
the source, which a missing or failing copy falls back to. A listing travels as one tarball of its
files. Every file of a mirrored corpus is checked against its pin, whichever way it arrived.
"""

import argparse
import gzip
import hashlib
import html.parser
import json
import struct
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from emblema.shared.ports.artifact_store import ArtifactStore

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
CHUNK = 8 * 1024 * 1024
NESTED_WRAPPER_LIMIT = 3
RANGE_NOT_SATISFIABLE = 416
# Seconds a connection may stay silent, to open or between two reads, before it is given up: a
# server that stalls mid-file would otherwise hold the process, and a paid accelerator, forever.
# What arrived stays in the partial file, so the next run resumes from it.
SILENCE = 60.0
DIGEST_CACHE = ".digests.json"
# The copies of open corpora in the artifact store, as `sha256sum` lines over `<corpus>/<file>`.
MIRROR_PINS = Path(__file__).resolve().parent / "raw_mirror.sha256"
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
        filename: Local name under ``data/raw/<corpus>/``; may sit in a subdirectory.
        md5: Publisher's checksum, when one is published as MD5; verified after download.
        sha256: Publisher's checksum, when one is published as SHA-256; verified after download.
    """

    url: str
    filename: str
    md5: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Listing:
    """A directory of files at the source of record, served as index pages rather than an archive.

    Every file the index names is downloaded under ``directory``, and the index pages it names are
    followed. The publisher states no checksum per file, so the listing is reported as one row:
    its size is the files' sizes summed, its digest the digest of every file's name and digest.

    Attributes:
        url: Index page at the source of record, ending in a slash.
        directory: Local directory under ``data/raw/<corpus>/`` the files land in.
    """

    url: str
    directory: str


@dataclass(frozen=True)
class Corpus:
    """A corpus and the archives that make it up.

    Attributes:
        key: Directory name under ``data/raw/`` and the command-line selector.
        title: Human-readable name with the publisher.
        archives: Files or listings to download, in order.
        mirrored: Whether its licence lets the files be copied to the artifact store.
    """

    key: str
    title: str
    archives: tuple[Archive | Listing, ...]
    mirrored: bool = False


# master of each repository on 2026-09-11; the archive URL below is immutable for a commit.
SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
OMNIANOMALY_COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
PHYSIONET = "https://physionet.org/files/challenge-2012/1.0.0"
PHYSIONET_2019 = "https://physionet.org/files/challenge-2019/1.0.0/training/"
ZENODO_ESA_AD_V2 = "https://zenodo.org/api/records/15237121/files"
ZENODO_OPSSAT_AD = "https://zenodo.org/api/records/12588359/files"
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"
# The collection's repository on 2025-06-19; a file resolved at a revision is immutable.
UTSD_REVISION = "7326ff5f4578da73d843fd675d760c6c6054017f"
UTSD = f"https://huggingface.co/datasets/thuml/UTSD/resolve/{UTSD_REVISION}"
# The shards of the volume read here with the digests the repository stores them under, in the
# format of `sha256sum`; a dash stands for a file the repository keeps outside its object store.
# The smaller volume waits for digests taken from downloaded files: the repository reports one
# digest for two of its shards of different sizes, so its listing cannot be trusted as published.
UTSD_SHARDS = Path(__file__).resolve().parent / "utsd_shards.sha256"


def utsd_archives(listing: Path = UTSD_SHARDS) -> tuple[Archive, ...]:
    """The collection's files, one archive per line of the pinned listing."""
    archives = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        digest, _, path = line.partition("  ")
        archives.append(
            Archive(url=f"{UTSD}/{path}", filename=path, sha256=None if digest == "-" else digest)
        )
    return tuple(archives)


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
        mirrored=True,
    ),
    Corpus(
        key="physionet2019",
        title="PhysioNet/CinC Challenge 2019 (physionet.org)",
        archives=(Listing(url=PHYSIONET_2019, directory="training"),),
        mirrored=True,
    ),
    Corpus(
        key="tep",
        title="Tennessee Eastman process simulation data (Harvard Dataverse 10.7910/DVN/6C3JR1)",
        # The training files alone: the testing files are the benchmark's evaluation data.
        archives=(
            Archive(
                url=f"{DATAVERSE}/3031241",
                filename="TEP_FaultFree_Training.RData",
                md5="ec126484534331f85001d8c4ebce6d17",
            ),
            Archive(
                url=f"{DATAVERSE}/3031242",
                filename="TEP_Faulty_Training.RData",
                md5="c5f594d54c47e620ff877feb58407fda",
            ),
        ),
    ),
    Corpus(
        key="utsd",
        title="UTSD, Unified Time Series Dataset, volume 12G (Hugging Face thuml/UTSD)",
        archives=utsd_archives(),
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
    Corpus(
        key="opssat_ad",
        title="OPSSAT-AD, the OPS-SAT anomaly detection benchmark (Zenodo 10.5281/zenodo.12588359)",
        archives=(
            Archive(
                url=f"{ZENODO_OPSSAT_AD}/segments.csv/content",
                filename="segments.csv",
                md5="1f4b6ffd0359efac96a6698ff557e0f9",
            ),
            Archive(
                url=f"{ZENODO_OPSSAT_AD}/dataset.csv/content",
                filename="dataset.csv",
                md5="c12fb83863afa4463dec07303a20bb14",
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
        verdict: Verdict against the publisher's checksum.
        error: Why the archive is missing, when it is; None once it is on disk.
    """

    corpus: str
    filename: str
    size: int
    sha256: str
    verdict: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.verdict.startswith("MISMATCH")

    @property
    def problem(self) -> str:
        return self.error or self.verdict


class Digests(TypedDict):
    size: int
    sha256: str
    md5: str


def note(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def download(url: str, target: Path, *, quiet: bool = False) -> None:
    """Download ``url`` to ``target``, resuming a partial file if one is left over."""
    if target.exists():
        if not quiet:
            note(f"{target.name}: already downloaded")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "emblema-fetch-corpora"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request, timeout=SILENCE) as response:
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


class _IndexLinks(html.parser.HTMLParser):
    """The targets of the anchors of an index page, as the page lists them."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def listed(url: str) -> Iterator[str]:
    """Every file an index page names, as a path relative to it, following the pages it names.

    A link that leaves the directory — upwards, to a query, to another host — names no file of it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "emblema-fetch-corpora"})
    with urllib.request.urlopen(request, timeout=SILENCE) as response:
        page = b"".join(iter(lambda: response.read(CHUNK), b"")).decode("utf-8", errors="replace")
    parser = _IndexLinks()
    parser.feed(page)
    for link in parser.links:
        if link.startswith(("?", "#")) or "://" in link:
            continue
        parts = PurePosixPath(urllib.parse.unquote(link)).parts
        if not parts or parts[0] in ("/", ".") or ".." in parts:
            continue
        if link.endswith("/"):
            yield from (f"{link}{name}" for name in listed(urllib.parse.urljoin(url, link)))
        else:
            yield "/".join(parts)


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


def fetch(
    corpus: Corpus, *, recheck: bool = False, store: "ArtifactStore | None" = None
) -> Iterator[Fetched]:
    """Every archive of the corpus, from the store's pinned copy first when a store is given."""
    directory = DATA_ROOT / corpus.key
    directory.mkdir(parents=True, exist_ok=True)
    pins = mirror_pins() if corpus.mirrored else {}
    for archive in corpus.archives:
        name = archive.filename if isinstance(archive, Archive) else archive.directory
        try:
            if isinstance(archive, Listing):
                yield fetch_listing(
                    corpus,
                    archive,
                    directory,
                    recheck=recheck,
                    store=store,
                    pin=pins.get(f"{corpus.key}/{bundle_name(archive)}"),
                    contents_pin=pins.get(f"{corpus.key}/{archive.directory}/"),
                )
            else:
                pin = pins.get(f"{corpus.key}/{archive.filename}")
                yield fetch_archive(
                    corpus, archive, directory, recheck=recheck, store=store, pin=pin
                )
        except Exception as error:
            # One archive is one file of one corpus. The rest of the run is still worth doing, and
            # the checksums of what did arrive are still worth reporting.
            note(f"{name}: FAILED, {error}")
            yield Fetched(corpus.key, name, 0, "", "not verified", f"{error}")


def fetch_listing(
    corpus: Corpus,
    listing: Listing,
    directory: Path,
    *,
    recheck: bool,
    store: "ArtifactStore | None" = None,
    pin: str | None = None,
    contents_pin: str | None = None,
) -> Fetched:
    """Download every file of the listing; report the listing as one row.

    The files of a listing are small and many, so each is hashed as it is checked rather than
    remembered: the note of digests is kept for archives of gigabytes, and rewriting it once per
    file would cost more than hashing a few kilobytes. A listing unpacked from the store's tarball
    is reported from the files on disk, without asking the source what it lists; the pin of its
    contents, checked whichever way the files came, turns a file missing, cut short or left over
    into a mismatch rather than a quieter digest.
    """
    root = directory / listing.directory
    tarball = directory / bundle_name(listing)
    marker = directory / f".unpacked-{tarball.name}"
    if (
        store is not None
        and pin is not None
        and not marker.exists()
        and from_store(store, pin, tarball)
    ):
        unpack(tarball, directory)
    if marker.exists():
        paths = sorted(path.relative_to(root).as_posix() for path in listing_files(root))
        note(f"{listing.directory}: {len(paths):,} files from the mirror")
    else:
        paths = sorted(listed(listing.url))
        present = sum(1 for path in paths if (root / path).exists())
        note(f"{listing.directory}: {len(paths):,} files, {present:,} already downloaded")
        for path in paths:
            download(urllib.parse.urljoin(listing.url, path), root / path, quiet=True)
    size, digest = listing_contents(root, paths)
    verdict = "none published"
    if contents_pin is not None and digest != contents_pin:
        verdict = (
            f"MISMATCH with the mirror's pin (got {digest}); "
            f"remove {root} and its .unpacked marker, then fetch again"
        )
    return Fetched(corpus.key, listing.directory, size, digest, verdict)


def listing_contents(root: Path, paths: Iterable[str]) -> tuple[int, str]:
    """Bytes and digest of a listing: the digest of every file's digest and path, in order."""
    size = 0
    digest = hashlib.sha256()
    for path in paths:
        target = root / path
        sha256, _ = digests(target)
        size += target.stat().st_size
        digest.update(f"{sha256}  {path}\n".encode())
    return size, digest.hexdigest()


def fetch_archive(
    corpus: Corpus,
    archive: Archive,
    directory: Path,
    *,
    recheck: bool,
    store: "ArtifactStore | None" = None,
    pin: str | None = None,
) -> Fetched:
    target = directory / archive.filename
    mirrored = (
        store is not None
        and pin is not None
        and not target.exists()
        and from_store(store, pin, target)
    )
    if not mirrored:
        download(archive.url, target)
    sha256, md5 = checksums(target, recheck=recheck)
    verdict = (
        checksum_status(archive.md5, md5)
        if archive.sha256 is None
        else checksum_status(archive.sha256, sha256)
    )
    if pin is not None and sha256 != pin and not verdict.startswith("MISMATCH"):
        verdict = f"MISMATCH with the mirror's pin (got {sha256})"
    row = Fetched(corpus.key, archive.filename, target.stat().st_size, sha256, verdict)
    if not row.ok:
        forget(target)
        target.replace(target.with_suffix(target.suffix + ".bad"))
    elif archive.filename.endswith((".zip", ".tar.gz")):
        unpack(target, directory)
    return row


def bundle_name(listing: Listing) -> str:
    """The name the store's copy of a listing goes by: one tarball of its directory."""
    return f"{listing.directory}.tar.gz"


def listing_files(root: Path) -> list[Path]:
    """The files of a listing on disk, without notes, metadata or interrupted downloads."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix != ".part"
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )


def bundle(root: Path, target: Path) -> Path:
    """One gzipped tarball of a listing's files.

    Names, owners, modes and times are fixed and the members sorted, so the same files tar to the
    same bytes under the same zlib; the listing's identity is still the pin of its contents, not
    of the tarball, which another zlib may compress differently.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with (
        partial.open("wb") as sink,
        gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tarball,
    ):
        for path in listing_files(root):
            member = tarfile.TarInfo(f"{root.name}/{path.relative_to(root).as_posix()}")
            member.size = path.stat().st_size
            member.mode = 0o644
            with path.open("rb") as source:
                tarball.addfile(member, source)
    partial.replace(target)
    return target


def from_store(store: "ArtifactStore", pin: str, target: Path) -> bool:
    """Lay the store's copy pinned to ``pin`` at ``target``; False when it cannot be had.

    The store verifies the bytes against the pin before the file appears, so a copy that is
    missing, damaged or unreachable leaves nothing behind and the source of record is asked
    instead.
    """
    from emblema.shared.adapters.storage.layout import address_of
    from emblema.shared.kernel.checksums import Checksum, HashAlgorithm
    from emblema.shared.kernel.retention import Retention

    ref = address_of(Checksum(HashAlgorithm.SHA256, pin), Retention.DURABLE)
    try:
        store.get_file(ref, target)
    except Exception as error:
        # Whatever went wrong with the copy, the source still has the file and the pin still
        # checks what arrives from it.
        note(f"{target.name}: not in the mirror ({error}), from the source")
        return False
    note(f"{target.name}: from the mirror")
    return True


def mirror_pins(path: Path | None = None) -> dict[str, str]:
    """The digest each mirrored copy must hash to, by ``<corpus>/<file>``; none before a mirror."""
    pins = MIRROR_PINS if path is None else path
    if not pins.exists():
        return {}
    entries = {}
    for line in pins.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        entries[name] = digest
    return entries


def write_mirror_pins(entries: dict[str, str], path: Path | None = None) -> None:
    pins = MIRROR_PINS if path is None else path
    lines = "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))
    pins.write_text(lines, encoding="utf-8")


def mirror(corpus: Corpus, store: "ArtifactStore") -> dict[str, str]:
    """Copy a corpus's files to the store and return the pins of the copies.

    The files are fetched and checked first, against the pins already written too, so a copy is
    only ever made of bytes that passed; a corpus mirrored for the first time is trusted as its
    source serves it, since no publisher checksum exists to check it by. A listing is tarred
    from the files on disk only when they are exactly the files the source lists, and pinned
    twice: by its contents, which a later mirror must match, and by its tarball, which the store
    addresses.

    Raises:
        SystemExit: If the corpus is not one whose licence lets it be copied, a file did not
            pass its checks, or the files on disk are not the listing's.
    """
    if not corpus.mirrored:
        raise SystemExit(f"{corpus.key} is not marked for the mirror: its licence was not cleared")
    rows = list(fetch(corpus))
    failed = [row for row in rows if not row.ok]
    if failed:
        raise SystemExit(f"not mirrored, fetch {corpus.key} again first:\n{problems(failed)}")
    directory = DATA_ROOT / corpus.key
    entries: dict[str, str] = {}
    for archive, row in zip(corpus.archives, rows, strict=True):
        if isinstance(archive, Listing):
            root = directory / archive.directory
            on_disk = [path.relative_to(root).as_posix() for path in listing_files(root)]
            if listing_contents(root, on_disk)[1] != row.sha256:
                raise SystemExit(f"{root} holds files the listing does not; remove them first")
            entries[f"{corpus.key}/{archive.directory}/"] = row.sha256
            source = bundle(root, directory / bundle_name(archive))
            key = f"{corpus.key}/{source.name}"
        else:
            source = directory / archive.filename
            key = f"{corpus.key}/{archive.filename}"
        digest = store.put_file(source).checksum.digest
        note(f"{key}: mirrored as {digest}")
        entries[key] = digest
    return entries


def report(rows: Iterable[Fetched]) -> str:
    lines = [
        "| Corpus | File | Bytes | SHA-256 | Publisher checksum |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row.corpus} | {row.filename} | {row.size:,} | `{row.sha256}` | {row.verdict} |"
        for row in rows
        if row.error is None
    )
    return "\n".join(lines)


def problems(rows: Iterable[Fetched]) -> str:
    lines = ["Not done, run again (a file that failed its published checksum is set aside):"]
    lines.extend(f"- {row.corpus}/{row.filename}: {row.problem}" for row in rows if not row.ok)
    return "\n".join(lines)


def configured_store_from_environment() -> "ArtifactStore":
    """The artifact store the environment names; the package is needed for the mirror alone."""
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store

    return configured_store(Settings())


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
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="copy the corpora to the configured artifact store and pin the copies; "
        "default: every corpus marked for the mirror",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="read each pinned copy from the configured artifact store before the source",
    )
    args = parser.parse_args()
    if args.list:
        for corpus in CORPORA:
            mirrored = ", mirrored" if corpus.mirrored else ""
            print(f"{corpus.key:14} {corpus.title}{mirrored}")
            for archive in corpus.archives:
                print(f"{'':14} {archive.url}")  # a listing's one index page, a file's address
        return
    if args.mirror:
        target = configured_store_from_environment()
        corpora = select(args.corpora) if args.corpora else tuple(c for c in CORPORA if c.mirrored)
        entries = mirror_pins()
        for corpus in corpora:
            entries.update(mirror(corpus, target))
        write_mirror_pins(entries)
        print(MIRROR_PINS.read_text(encoding="utf-8"), end="")
        return
    store = configured_store_from_environment() if args.from_store else None
    rows = [
        row
        for corpus in select(args.corpora)
        for row in fetch(corpus, recheck=args.recheck, store=store)
    ]
    print(f"\nData root: {DATA_ROOT}\n")
    print(report(rows))
    failed = [row for row in rows if not row.ok]
    if failed:
        print(f"\n{problems(rows)}")
        raise SystemExit(f"{len(failed)} of {len(rows)} archives need another run")


if __name__ == "__main__":
    main()
