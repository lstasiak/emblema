import hashlib
import io
import json
import os
import re
import struct
import tarfile
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Collection
from email.message import Message
from pathlib import Path
from urllib.parse import urlparse

import pytest

from emblema.shared.adapters.storage.layout import address_of
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm
from emblema.shared.kernel.retention import Retention
from scripts import fetch_corpora
from scripts.corpus_budget_report import DEFAULT_CONFIG, Budget
from scripts.fetch_corpora import (
    CORPORA,
    DEFLATE64,
    UTSD,
    Archive,
    Corpus,
    Fetched,
    Listing,
    bundle,
    bundle_name,
    checksum_status,
    checksums,
    download,
    extract,
    fetch,
    listed,
    mirror,
    mirror_pins,
    problems,
    report,
    select,
    unpack,
    utsd_archives,
    write_mirror_pins,
)

LOCAL_HEADER = "<4sHHHHHIIIHH"
CENTRAL_HEADER = "<4sHHHHHHIIIHHHHHII"
END_OF_DIRECTORY = "<4sHHHHIIH"


def zip_with(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


def handmade_zip(
    path: Path,
    members: dict[str, bytes],
    *,
    deflate64: Collection[str] = (),
    corrupt: Collection[str] = (),
) -> Path:
    """A zip written by hand, because ``zipfile`` can neither write nor read Deflate64.

    Members named in ``deflate64`` use method 9, the rest are stored; a member named in
    ``corrupt`` gets a CRC that does not match its bytes.
    """
    import inflate64

    entries, directory, offset = bytearray(), bytearray(), 0
    for name, payload in members.items():
        if name in deflate64:
            deflater = inflate64.Deflater()
            body = deflater.deflate(payload) + deflater.flush()
        else:
            body = payload
        method = DEFLATE64 if name in deflate64 else 0
        crc = zlib.crc32(b"not the payload" if name in corrupt else payload)
        encoded = name.encode()
        sizes = (crc, len(body), len(payload), len(encoded))
        entries += struct.pack(LOCAL_HEADER, b"PK\x03\x04", 20, 0, method, 0, 0, *sizes, 0)
        entries += encoded + body
        directory += struct.pack(
            CENTRAL_HEADER, b"PK\x01\x02", 20, 20, 0, method, 0, 0, *sizes, 0, 0, 0, 0, 0, offset
        )
        directory += encoded
        offset = len(entries)
    end = struct.pack(
        END_OF_DIRECTORY,
        b"PK\x05\x06",
        0,
        0,
        len(members),
        len(members),
        len(directory),
        offset,
        0,
    )
    path.write_bytes(bytes(entries + directory + end))
    return path


def count_hashing(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the hashing with a counter: what it returns does not matter, how often does."""
    hashed: list[Path] = []

    def digests(path: Path) -> tuple[str, str]:
        hashed.append(path)
        return "sha", "md5"

    monkeypatch.setattr(fetch_corpora, "digests", digests)
    return hashed


class FakeResponse:
    """What ``urlopen`` returns, reduced to the surface ``download`` touches."""

    def __init__(self, status: int, payload: bytes, *, stalls_after: int | None = None) -> None:
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}
        self._buffer = io.BytesIO(payload)
        self._stalls_after = stalls_after

    def read(self, size: int) -> bytes:
        if self._stalls_after is not None and self._buffer.tell() >= self._stalls_after:
            raise TimeoutError("the server went silent")
        if self._stalls_after is not None:
            size = min(size, self._stalls_after - self._buffer.tell())
        return self._buffer.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def serve(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        assert timeout == fetch_corpora.SILENCE
        requests.append(request)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requests


def test_download_resumes_a_partial_file_when_the_server_honours_the_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.zip"
    target.with_suffix(".zip.part").write_bytes(b"abc")
    requests = serve(monkeypatch, FakeResponse(206, b"def"))

    download("https://example.org/file.zip", target)

    assert target.read_bytes() == b"abcdef"
    assert not target.with_suffix(".zip.part").exists()
    assert requests[0].get_header("Range") == "bytes=3-"


def test_download_starts_over_when_the_server_ignores_the_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.zip"
    target.with_suffix(".zip.part").write_bytes(b"abc")
    serve(monkeypatch, FakeResponse(200, b"xyz"))

    download("https://example.org/file.zip", target)

    assert target.read_bytes() == b"xyz"


def test_download_treats_an_unsatisfiable_range_as_a_complete_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.zip"
    target.with_suffix(".zip.part").write_bytes(b"complete")

    def refuse(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            "https://example.org", 416, "Range Not Satisfiable", Message(), None
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    download("https://example.org/file.zip", target)

    assert target.read_bytes() == b"complete"


def test_a_server_that_goes_silent_mid_file_ends_the_download_and_leaves_what_arrived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.zip"
    serve(monkeypatch, FakeResponse(200, b"abcdef", stalls_after=4))

    with pytest.raises(TimeoutError):
        download("https://example.org/file.zip", target)

    assert not target.exists()
    assert target.with_suffix(".zip.part").read_bytes() == b"abcd"


def test_download_skips_a_file_that_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "file.zip"
    target.write_bytes(b"kept")
    requests = serve(monkeypatch, FakeResponse(200, b"new"))

    download("https://example.org/file.zip", target)

    assert target.read_bytes() == b"kept"
    assert requests == []


def test_checksum_status_reports_match_mismatch_and_absence():
    assert checksum_status(None, "abc") == "none published"
    assert checksum_status("abc", "abc") == "ok"
    assert checksum_status("abc", "def") == "MISMATCH (got def)"


def test_a_thin_wrapper_zip_has_its_inner_zip_extracted(tmp_path: Path):
    inner = zip_with(tmp_path / "inner.zip", {"train_FD001.txt": b"1 1 0 0 0\n"})
    outer = zip_with(tmp_path / "outer.zip", {"wrapper/CMAPSSData.zip": inner.read_bytes()})
    into = tmp_path / "out"

    unpack(outer, into)

    assert (into / "wrapper" / "CMAPSSData" / "train_FD001.txt").read_bytes() == b"1 1 0 0 0\n"


def test_finder_metadata_does_not_count_towards_the_wrapper_limit(tmp_path: Path):
    inner = zip_with(tmp_path / "inner.zip", {"train_FD001.txt": b"1 1 0 0 0\n"})
    outer = zip_with(
        tmp_path / "outer.zip",
        # Three files is the whole wrapper; the Finder entries would push it over the limit and
        # leave the corpus packed.
        {
            "wrapper/CMAPSSData.zip": inner.read_bytes(),
            "wrapper/readme.txt": b"engine degradation\n",
            "wrapper/Damage Propagation Modeling.pdf": b"%PDF-1.4\n",
            "__MACOSX/wrapper/._CMAPSSData.zip": b"\x00\x05\x16\x07 resource fork",
            "__MACOSX/wrapper/._readme.txt": b"\x00\x05\x16\x07 resource fork",
        },
    )
    into = tmp_path / "out"

    unpack(outer, into)

    assert (into / "wrapper" / "CMAPSSData" / "train_FD001.txt").exists()


def test_a_resource_fork_named_like_a_zip_is_not_unpacked(tmp_path: Path):
    inner = zip_with(tmp_path / "inner.zip", {"train_FD001.txt": b"1 1 0 0 0\n"})
    outer = zip_with(
        tmp_path / "outer.zip",
        # A resource fork carries the bytes of the file it shadows, so it can read as a zip.
        {"wrapper/CMAPSSData.zip": inner.read_bytes(), "wrapper/._readme.zip": inner.read_bytes()},
    )
    into = tmp_path / "out"

    unpack(outer, into)

    assert (into / "wrapper" / "CMAPSSData" / "train_FD001.txt").exists()
    assert not (into / "wrapper" / "._readme").exists()


def test_an_archive_of_many_zips_is_data_and_stays_compressed(tmp_path: Path):
    channel = zip_with(tmp_path / "channel.zip", {"channel_1": b"pickle"}).read_bytes()
    members = {f"channels/channel_{i}.zip": channel for i in range(4)}
    members["channels.csv"] = b"Channel,Target\n"
    mission = zip_with(tmp_path / "mission.zip", members)
    into = tmp_path / "out"

    unpack(mission, into)

    assert sorted(p.name for p in (into / "channels").iterdir()) == [
        f"channel_{i}.zip" for i in range(4)
    ]


def test_unpack_runs_once_per_archive(tmp_path: Path):
    archive = zip_with(tmp_path / "a.zip", {"file.txt": b"x"})
    into = tmp_path / "out"
    unpack(archive, into)
    (into / "file.txt").write_bytes(b"edited")

    unpack(archive, into)

    assert (into / "file.txt").read_bytes() == b"edited"


def test_extract_handles_gzipped_tarballs(tmp_path: Path):
    archive = tmp_path / "set-c.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("set-c/1.txt")
        info.size = 5
        bundle.addfile(info, io.BytesIO(b"hello"))

    extracted = list(extract(archive, tmp_path / "out"))

    assert extracted == [tmp_path / "out" / "set-c" / "1.txt"]
    assert extracted[0].read_bytes() == b"hello"


def test_fetch_sets_aside_a_file_that_fails_its_published_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"data"))
    corpus = Corpus(
        key="x",
        title="x",
        archives=(Archive(url="https://example.org/a.zip", filename="a.zip", md5="0" * 32),),
    )

    (row,) = fetch(corpus)

    assert not row.ok
    assert not (tmp_path / "x" / "a.zip").exists()
    assert (tmp_path / "x" / "a.zip.bad").read_bytes() == b"data"
    assert not (tmp_path / "x" / ".unpacked-a.zip").exists()


def test_fetch_unpacks_a_file_whose_checksum_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = zip_with(tmp_path / "payload.zip", {"file.txt": b"x"}).read_bytes()
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(payload))
    corpus = Corpus(
        key="x", title="x", archives=(Archive(url="https://example.org/a.zip", filename="a.zip"),)
    )

    (row,) = fetch(corpus)

    assert (row.ok, row.verdict) == (True, "none published")
    assert (tmp_path / "x" / "file.txt").read_bytes() == b"x"


COMMIT = re.compile(r"[0-9a-f]{40}")


@pytest.mark.parametrize("corpus", CORPORA, ids=lambda corpus: corpus.key)
def test_the_budget_file_names_the_source_the_fetcher_downloads_from(corpus: Corpus):
    source = Budget.load(DEFAULT_CONFIG).corpora[corpus.key].source
    urls = [archive.url for archive in corpus.archives]

    assert any(urlparse(source).netloc == urlparse(url).netloc for url in urls)
    assert {sha for url in urls for sha in COMMIT.findall(url)} <= set(COMMIT.findall(source))


def test_select_defaults_to_every_corpus_and_rejects_unknown_keys():
    assert select([]) == CORPORA
    assert [corpus.key for corpus in select(["skab", "cmapss"])] == ["skab", "cmapss"]
    with pytest.raises(SystemExit, match="unknown corpus: nope"):
        select(["nope"])


def test_a_deflate64_member_is_extracted_where_zipfile_gives_up(tmp_path: Path):
    table = b"Channel,Target\n" + b"channel_1,NO\n" * 500
    archive = handmade_zip(
        tmp_path / "mission.zip",
        {"ESA-Mission1/channels.csv": table, "ESA-Mission1/channels/channel_1.zip": b"stored"},
        deflate64=["ESA-Mission1/channels.csv"],
    )
    into = tmp_path / "out"

    with pytest.raises(NotImplementedError):
        zipfile.ZipFile(archive).extractall(tmp_path / "zipfile-says-no")
    extracted = list(extract(archive, into))

    assert extracted == [
        into / "ESA-Mission1" / "channels.csv",
        into / "ESA-Mission1" / "channels" / "channel_1.zip",
    ]
    assert (into / "ESA-Mission1" / "channels.csv").read_bytes() == table
    assert (into / "ESA-Mission1" / "channels" / "channel_1.zip").read_bytes() == b"stored"


def test_a_deflate64_member_that_decompresses_wrong_is_refused(tmp_path: Path):
    archive = handmade_zip(
        tmp_path / "mission.zip",
        {"channels.csv": b"Channel,Target\n"},
        deflate64=["channels.csv"],
        corrupt=["channels.csv"],
    )

    with pytest.raises(ValueError, match=r"channels\.csv is corrupt"):
        list(extract(archive, tmp_path / "out"))


def test_checksums_are_remembered_so_a_second_run_hashes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "big.zip"
    target.write_bytes(b"payload")
    hashed = count_hashing(monkeypatch)

    first = checksums(target)

    assert checksums(target) == first == ("sha", "md5")
    assert hashed == [target]
    assert json.loads((tmp_path / ".digests.json").read_text())["big.zip"]["size"] == 7


def test_a_file_of_another_size_is_hashed_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "big.zip"
    target.write_bytes(b"payload")
    hashed = count_hashing(monkeypatch)
    checksums(target)

    target.write_bytes(b"a longer payload")
    checksums(target)

    assert hashed == [target, target]


def test_recheck_distrusts_what_was_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "big.zip"
    target.write_bytes(b"payload")
    hashed = count_hashing(monkeypatch)
    checksums(target)

    checksums(target, recheck=True)

    assert hashed == [target, target]


def test_an_unreadable_note_is_no_note_at_all(tmp_path: Path):
    target = tmp_path / "file.txt"
    target.write_bytes(b"payload")
    (tmp_path / ".digests.json").write_text("{ truncated")

    assert checksums(target) == fetch_corpora.digests(target)


def test_a_file_set_aside_is_forgotten_so_its_replacement_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"data"))
    corpus = Corpus(
        key="x",
        title="x",
        archives=(Archive(url="https://example.org/a.zip", filename="a.zip", md5="0" * 32),),
    )

    (row,) = fetch(corpus)

    assert not row.ok
    assert "a.zip" not in json.loads((tmp_path / "x" / ".digests.json").read_text())


def test_an_archive_that_fails_does_not_stop_the_ones_behind_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)

    def flaky(url: str, target: Path) -> None:
        if target.name == "a.zip":
            raise urllib.error.URLError("connection reset")
        target.write_bytes(b"data")

    monkeypatch.setattr(fetch_corpora, "download", flaky)
    corpus = Corpus(
        key="x",
        title="x",
        archives=(
            Archive(url="https://example.org/a.zip", filename="a.zip"),
            Archive(url="https://example.org/b.txt", filename="b.txt"),
        ),
    )

    first, second = fetch(corpus)

    assert not first.ok
    assert "connection reset" in first.problem
    assert second.ok
    assert (tmp_path / "x" / "b.txt").read_bytes() == b"data"


def test_the_report_holds_what_arrived_and_the_problems_what_did_not():
    rows = [
        Fetched("x", "a.zip", 4, "abc", "ok"),
        Fetched("x", "b.zip", 0, "", "not verified", "connection reset"),
        Fetched("x", "c.zip", 4, "def", "MISMATCH (got def)"),
    ]

    table, listing = report(rows), problems(rows)

    assert "a.zip" in table
    assert "b.zip" not in table
    assert "- x/b.zip: connection reset" in listing
    assert "- x/c.zip: MISMATCH (got def)" in listing
    assert "a.zip" not in listing


def test_download_lays_a_file_in_a_directory_that_does_not_exist_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "UTSD-12G" / "data-00000-of-00080.arrow"
    serve(monkeypatch, FakeResponse(200, b"shard"))

    download("https://example.org/data-00000-of-00080.arrow", target)

    assert target.read_bytes() == b"shard"


def test_the_pinned_listing_names_every_shard_of_the_collection_at_one_revision():
    archives = utsd_archives()

    assert len(archives) == 82
    assert all(archive.url == f"{UTSD}/{archive.filename}" for archive in archives)
    assert {archive.filename.split("/")[0] for archive in archives} == {"UTSD-12G"}
    shards = [archive for archive in archives if archive.filename.endswith(".arrow")]
    assert len(shards) == 80
    digests = [archive.sha256 for archive in shards]
    assert all(digest and len(digest) == 64 for digest in digests)
    # Two shards of different sizes cannot hash equal; a digest listed twice was copied wrong.
    assert len(set(digests)) == len(digests)
    # The repository keeps its two small JSON files outside the object store: no digest to check.
    assert all(archive.sha256 is None for archive in archives if archive.filename.endswith(".json"))


def test_a_listing_is_read_from_a_file_of_digests_and_paths(tmp_path: Path):
    listing = tmp_path / "shards.sha256"
    listing.write_text("abc  V/a.arrow\n-  V/state.json\n", encoding="utf-8")

    assert utsd_archives(listing) == (
        Archive(url=f"{UTSD}/V/a.arrow", filename="V/a.arrow", sha256="abc"),
        Archive(url=f"{UTSD}/V/state.json", filename="V/state.json"),
    )


def test_fetch_verifies_a_published_sha256_and_sets_a_mismatch_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"data"))
    right = hashlib.sha256(b"data").hexdigest()
    corpus = Corpus(
        key="x",
        title="x",
        archives=(
            Archive(url="https://example.org/a.arrow", filename="a.arrow", sha256=right),
            Archive(url="https://example.org/b.arrow", filename="b.arrow", sha256="0" * 64),
        ),
    )

    ok, bad = fetch(corpus)

    assert (ok.ok, ok.verdict) == (True, "ok")
    assert (bad.ok, bad.verdict.startswith("MISMATCH")) == (False, True)
    assert (tmp_path / "x" / "b.arrow.bad").exists()


PAGES = {
    "https://example.org/training/": (
        '<a href="../">../</a><a href="?C=N;O=D">Name</a><a href="/">root</a>'
        '<a href="training_setA/">training_setA/</a><a href="training_setB/">training_setB/</a>'
        '<a href="https://example.org/elsewhere">away</a><a href="x/../escape.psv">escape</a>'
    ),
    "https://example.org/training/training_setA/": (
        '<a href="../">../</a><a href="p1.psv">p1</a><a href="p%204.psv">p 4</a>'
    ),
    "https://example.org/training/training_setB/": '<a href="p2.psv">p2</a><a href="p3.psv">p3</a>',
}


def test_an_index_page_lists_its_files_and_follows_the_pages_it_names(
    monkeypatch: pytest.MonkeyPatch,
):
    def urlopen(request: urllib.request.Request, *, timeout: float) -> FakeResponse:
        return FakeResponse(200, PAGES[request.full_url].encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert sorted(listed("https://example.org/training/")) == [
        "training_setA/p 4.psv",
        "training_setA/p1.psv",
        "training_setB/p2.psv",
        "training_setB/p3.psv",
    ]


def test_fetch_downloads_every_file_of_a_listing_and_reports_it_as_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        fetch_corpora, "listed", lambda url: iter(["training_setB/p2.psv", "training_setA/p1.psv"])
    )
    fetched: list[str] = []

    def download(url: str, target: Path, *, quiet: bool = False) -> None:
        fetched.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(url.encode())

    monkeypatch.setattr(fetch_corpora, "download", download)
    remembered = tmp_path / "x" / "training" / "training_setA" / ".digests.json"
    corpus = Corpus(
        key="x",
        title="x",
        archives=(Listing(url="https://example.org/training/", directory="training"),),
    )

    (row,) = fetch(corpus)

    assert fetched == [
        "https://example.org/training/training_setA/p1.psv",
        "https://example.org/training/training_setB/p2.psv",
    ]
    assert (tmp_path / "x" / "training" / "training_setA" / "p1.psv").exists()
    # Small files are hashed, not remembered: no note of digests beside them.
    assert not remembered.exists()
    assert (row.filename, row.verdict, row.ok) == ("training", "none published", True)
    assert row.size == sum(len(url) for url in fetched)
    digest = hashlib.sha256()
    for path in ("training_setA/p1.psv", "training_setB/p2.psv"):
        content = f"https://example.org/training/{path}".encode()
        digest.update(f"{hashlib.sha256(content).hexdigest()}  {path}\n".encode())
    assert row.sha256 == digest.hexdigest()


def sha256_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stored(store: LocalDirectoryArtifactStore, content: bytes) -> str:
    return store.put(content).checksum.digest


def refuse_the_source(*args: object, **kwargs: object) -> None:
    raise AssertionError("the source of record was asked")


@pytest.fixture
def mirrored_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(fetch_corpora, "DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(fetch_corpora, "MIRROR_PINS", tmp_path / "pins.sha256")
    return tmp_path


def archive_corpus(*, mirrored: bool = True) -> Corpus:
    return Corpus(
        key="x",
        title="x",
        archives=(Archive(url="https://example.org/a.txt", filename="a.txt"),),
        mirrored=mirrored,
    )


def listing_corpus() -> Corpus:
    return Corpus(
        key="x",
        title="x",
        archives=(Listing(url="https://example.org/training/", directory="training"),),
        mirrored=True,
    )


def lay_listing(root: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)


def test_a_pinned_copy_is_read_from_the_store_without_asking_the_source(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    store = LocalDirectoryArtifactStore(mirrored_root / "store")
    write_mirror_pins({"x/a.txt": stored(store, b"data")})
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: None)

    (row,) = fetch(archive_corpus(), store=store)

    assert (row.ok, row.sha256) == (True, sha256_of(b"data"))
    assert (mirrored_root / "raw" / "x" / "a.txt").read_bytes() == b"data"


def test_a_copy_missing_from_the_store_falls_back_to_the_source(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    store = LocalDirectoryArtifactStore(mirrored_root / "store")
    write_mirror_pins({"x/a.txt": sha256_of(b"data")})
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"data"))

    (row,) = fetch(archive_corpus(), store=store)

    assert (row.ok, row.sha256) == (True, sha256_of(b"data"))


def test_a_file_from_the_source_that_misses_its_pin_is_set_aside(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    write_mirror_pins({"x/a.txt": sha256_of(b"data")})
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"other"))

    (row,) = fetch(archive_corpus())

    assert not row.ok
    assert row.verdict.startswith("MISMATCH with the mirror's pin")
    assert (mirrored_root / "raw" / "x" / "a.txt.bad").read_bytes() == b"other"


def test_a_corpus_not_marked_for_the_mirror_ignores_the_pins(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    write_mirror_pins({"x/a.txt": sha256_of(b"data")})
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"other"))

    (row,) = fetch(archive_corpus(mirrored=False))

    assert row.ok


def test_the_same_files_tar_to_the_same_bytes_whatever_their_times_and_order(tmp_path: Path):
    files = {"training_setA/p1.psv": b"1", "training_setB/p2.psv": b"22"}
    lay_listing(tmp_path / "one" / "training", files)
    lay_listing(tmp_path / "two" / "training", dict(reversed(files.items())))
    (tmp_path / "two" / "training" / ".DS_Store").write_bytes(b"finder")
    os.utime(tmp_path / "two" / "training" / "training_setA" / "p1.psv", (0, 0))

    first = bundle(tmp_path / "one" / "training", tmp_path / "one.tar.gz")
    second = bundle(tmp_path / "two" / "training", tmp_path / "two.tar.gz")

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as tarball:
        assert tarball.getnames() == [
            "training/training_setA/p1.psv",
            "training/training_setB/p2.psv",
        ]


def test_a_mirrored_listing_unpacks_from_the_store_and_reports_as_the_source_would(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    files = {"training_setA/p1.psv": b"1", "training_setB/p2.psv": b"22"}
    lay_listing(mirrored_root / "source" / "training", files)
    store = LocalDirectoryArtifactStore(mirrored_root / "store")
    tarball = bundle(mirrored_root / "source" / "training", mirrored_root / "training.tar.gz")
    write_mirror_pins({"x/training.tar.gz": store.put_file(tarball).checksum.digest})
    monkeypatch.setattr(fetch_corpora, "listed", refuse_the_source)
    monkeypatch.setattr(fetch_corpora, "download", refuse_the_source)

    (row,) = fetch(listing_corpus(), store=store)
    (again,) = fetch(listing_corpus(), store=store)

    root = mirrored_root / "raw" / "x" / "training"
    assert {path: (root / path).read_bytes() for path in files} == files
    assert (row.ok, row.size, row.sha256) == (True, 3, contents_of(files))
    assert again == row


def contents_of(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        digest.update(f"{sha256_of(content)}  {path}\n".encode())
    return digest.hexdigest()


def unpacked_listing(mirrored_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A listing of two files laid from the store's tarball, both its pins written."""
    files = {"training_setA/p1.psv": b"1", "training_setB/p2.psv": b"22"}
    lay_listing(mirrored_root / "source" / "training", files)
    store = LocalDirectoryArtifactStore(mirrored_root / "store")
    tarball = bundle(mirrored_root / "source" / "training", mirrored_root / "training.tar.gz")
    write_mirror_pins(
        {
            "x/training.tar.gz": store.put_file(tarball).checksum.digest,
            "x/training/": contents_of(files),
        }
    )
    monkeypatch.setattr(fetch_corpora, "listed", refuse_the_source)
    monkeypatch.setattr(fetch_corpora, "download", refuse_the_source)
    (row,) = fetch(listing_corpus(), store=store)
    assert row.ok
    return mirrored_root / "raw" / "x" / "training"


def test_a_file_of_a_listing_lost_after_it_was_unpacked_is_a_mismatch(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    root = unpacked_listing(mirrored_root, monkeypatch)
    (root / "training_setA" / "p1.psv").unlink()

    (row,) = fetch(listing_corpus())

    assert not row.ok
    assert row.verdict.startswith("MISMATCH with the mirror's pin")


def test_a_file_left_over_in_a_listing_is_a_mismatch_but_an_interrupted_download_is_not(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    root = unpacked_listing(mirrored_root, monkeypatch)
    (root / "training_setA" / "p9.psv.part").write_bytes(b"half")

    (with_partial,) = fetch(listing_corpus())
    (root / "training_setA" / "p9.psv").write_bytes(b"stray")
    (with_stray,) = fetch(listing_corpus())

    assert with_partial.ok
    assert not with_stray.ok


def test_a_listing_from_the_source_is_checked_against_the_pin_of_its_contents(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    write_mirror_pins({"x/training/": contents_of({"p1.psv": b"pinned"})})
    monkeypatch.setattr(fetch_corpora, "listed", lambda url: iter(["p1.psv"]))

    def download(url: str, target: Path, *, quiet: bool = False) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"served now")

    monkeypatch.setattr(fetch_corpora, "download", download)

    (row,) = fetch(listing_corpus())

    assert not row.ok
    assert row.verdict.startswith("MISMATCH with the mirror's pin")


def test_mirror_copies_each_file_and_tarball_and_returns_their_pins(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    corpus = Corpus(
        key="x",
        title="x",
        archives=(
            Archive(url="https://example.org/a.txt", filename="a.txt"),
            Listing(url="https://example.org/training/", directory="training"),
        ),
        mirrored=True,
    )
    lay_listing(mirrored_root / "raw" / "x" / "training", {"p1.psv": b"1"})
    monkeypatch.setattr(fetch_corpora, "listed", lambda url: iter(["p1.psv"]))
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target, quiet=False: None)
    (mirrored_root / "raw" / "x" / "a.txt").write_bytes(b"data")
    store = LocalDirectoryArtifactStore(mirrored_root / "store")

    pins = mirror(corpus, store)

    tarball = mirrored_root / "raw" / "x" / "training.tar.gz"
    assert pins == {
        "x/a.txt": sha256_of(b"data"),
        "x/training.tar.gz": sha256_of(tarball.read_bytes()),
        "x/training/": contents_of({"p1.psv": b"1"}),
    }
    for key in ("x/a.txt", "x/training.tar.gz"):
        digest = pins[key]
        assert store.exists(address_of(Checksum(HashAlgorithm.SHA256, digest), Retention.DURABLE))


def test_mirror_refuses_a_corpus_whose_licence_was_not_cleared(mirrored_root: Path):
    store = LocalDirectoryArtifactStore(mirrored_root / "store")

    with pytest.raises(SystemExit, match="not marked for the mirror"):
        mirror(archive_corpus(mirrored=False), store)


def test_mirror_refuses_a_file_that_did_not_pass_its_checks(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    write_mirror_pins({"x/a.txt": sha256_of(b"data")})
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target: target.write_bytes(b"other"))
    store = LocalDirectoryArtifactStore(mirrored_root / "store")

    with pytest.raises(SystemExit, match="fetch x again"):
        mirror(archive_corpus(), store)
    assert not (mirrored_root / "store").exists()


def test_mirror_refuses_a_listing_that_changed_since_it_was_pinned(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    lay_listing(mirrored_root / "raw" / "x" / "training", {"p1.psv": b"changed"})
    monkeypatch.setattr(fetch_corpora, "listed", lambda url: iter(["p1.psv"]))
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target, quiet=False: None)
    write_mirror_pins({"x/training/": contents_of({"p1.psv": b"pinned before"})})
    store = LocalDirectoryArtifactStore(mirrored_root / "store")

    with pytest.raises(SystemExit, match="fetch x again"):
        mirror(listing_corpus(), store)
    assert not (mirrored_root / "store").exists()


def test_mirror_refuses_to_tar_files_the_listing_does_not_hold(
    mirrored_root: Path, monkeypatch: pytest.MonkeyPatch
):
    lay_listing(mirrored_root / "raw" / "x" / "training", {"p1.psv": b"1", "mine.psv": b"2"})
    monkeypatch.setattr(fetch_corpora, "listed", lambda url: iter(["p1.psv"]))
    monkeypatch.setattr(fetch_corpora, "download", lambda url, target, quiet=False: None)
    store = LocalDirectoryArtifactStore(mirrored_root / "store")

    with pytest.raises(SystemExit, match="files the listing does not"):
        mirror(listing_corpus(), store)
    assert not (mirrored_root / "store").exists()


def test_the_pins_name_only_files_of_corpora_marked_for_the_mirror():
    names = {
        name
        for corpus in CORPORA
        if corpus.mirrored
        for archive in corpus.archives
        for name in (
            (f"{corpus.key}/{archive.filename}",)
            if isinstance(archive, Archive)
            else (f"{corpus.key}/{bundle_name(archive)}", f"{corpus.key}/{archive.directory}/")
        )
    }

    assert set(mirror_pins()) <= names


def test_the_pins_round_trip_through_their_file(tmp_path: Path):
    pins = {"b/x.zip": "1" * 64, "a/training.tar.gz": "2" * 64}

    write_mirror_pins(pins, tmp_path / "pins")

    assert mirror_pins(tmp_path / "pins") == pins
    assert (tmp_path / "pins").read_text().splitlines()[0].endswith("a/training.tar.gz")
