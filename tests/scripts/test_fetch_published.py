"""What a run reads is fetched from the store once, and only what the store holds."""

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from scripts.fetch_published import fetch_blocks, fetch_labels


def labels_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        content = b"ID,Channel\n"
        info = tarfile.TarInfo("esa_ad/ESA-Mission1/labels.csv")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_the_labels_are_unpacked_where_the_tasks_read_them(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    ref = store.put(labels_archive(), Retention.TRANSIENT)

    held = fetch_labels(store, ref.checksum.digest, tmp_path / "raw")

    assert held == len(labels_archive())
    assert (tmp_path / "raw" / "esa_ad" / "ESA-Mission1" / "labels.csv").read_bytes() == (
        b"ID,Channel\n"
    )


def test_labels_the_store_does_not_hold_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        fetch_labels(InMemoryArtifactStore(), "a" * 64, tmp_path / "raw")


def test_each_publications_block_is_laid_in_the_workspace_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryArtifactStore()
    block = store.put(b"x" * 2**20)
    manifest_ref = store.put(b"a manifest")
    manifest = SimpleNamespace(corpus="esa_ad_raw", block=block)
    monkeypatch.setattr(PublishedCorpusManifestJson, "decode", lambda _self, _data: manifest)

    said = fetch_blocks(store, tmp_path, [manifest_ref.checksum.digest])

    assert said == ["esa_ad_raw: 0.0 GB"]
    assert (tmp_path / block.checksum.digest).read_bytes() == b"x" * 2**20


def test_a_publication_the_store_does_not_hold_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        fetch_blocks(InMemoryArtifactStore(), tmp_path, ["a" * 64])
