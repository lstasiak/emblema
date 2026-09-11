import io
import tarfile
import urllib.error
import urllib.request
import zipfile
from email.message import Message
from pathlib import Path

import pytest

from scripts import fetch_corpora
from scripts.fetch_corpora import (
    CORPORA,
    Archive,
    Corpus,
    checksum_status,
    download,
    extract,
    fetch,
    select,
    unpack,
)


def zip_with(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    return path


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
