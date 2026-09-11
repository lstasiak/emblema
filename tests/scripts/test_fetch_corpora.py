import io
import json
import struct
import tarfile
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Collection
from email.message import Message
from pathlib import Path

import pytest

from scripts import fetch_corpora
from scripts.fetch_corpora import (
    CORPORA,
    DEFLATE64,
    Archive,
    Corpus,
    Fetched,
    checksum_status,
    checksums,
    download,
    extract,
    fetch,
    problems,
    report,
    select,
    unpack,
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

    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}
        self._buffer = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def serve(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> list[urllib.request.Request]:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request) -> FakeResponse:
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

    def refuse(request: urllib.request.Request) -> FakeResponse:
        raise urllib.error.HTTPError(
            "https://example.org", 416, "Range Not Satisfiable", Message(), None
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)

    download("https://example.org/file.zip", target)

    assert target.read_bytes() == b"complete"


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


def test_finder_metadata_in_a_wrapper_is_neither_data_nor_a_nested_zip(tmp_path: Path):
    inner = zip_with(tmp_path / "inner.zip", {"train_FD001.txt": b"1 1 0 0 0\n"})
    outer = zip_with(
        tmp_path / "outer.zip",
        {
            "wrapper/CMAPSSData.zip": inner.read_bytes(),
            "__MACOSX/wrapper/._CMAPSSData.zip": b"\x00\x05\x16\x07 resource fork",
            "wrapper/._readme.zip": b"not a zip either",
        },
    )
    into = tmp_path / "out"

    unpack(outer, into)

    assert (into / "wrapper" / "CMAPSSData" / "train_FD001.txt").exists()


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

    assert (row.ok, row.md5_status) == (True, "none published")
    assert (tmp_path / "x" / "file.txt").read_bytes() == b"x"


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
