"""What the block reader does beyond the port's contract: its cache, and what it refuses."""

from pathlib import Path

import pytest

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)
from emblema.pretraining.domain.exceptions import (
    InvalidTrainingCorpusError,
    UnreadablePublishedCorpusError,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.support.published import EMPTY_UNIT, TRAINING_UNIT, VALIDATION_UNIT, manifest_of, publish


class RefusingStore(InMemoryArtifactStore):
    """A store that holds what was put but will not hand a file out: a network that is down."""

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        raise AssertionError("the block was fetched although it was already in the workspace")


def test_the_block_is_fetched_once_and_kept_under_its_digest(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    workspace = tmp_path / "reader"
    reader = BlockTrainingCorpusReader(store, workspace)

    first = reader.read(published.manifest)
    cached = workspace / published.block.checksum.digest
    assert cached.is_file(), "the block is kept where the Catalog's archive would keep it"

    again = BlockTrainingCorpusReader(_refusing(store), workspace).read(published.manifest)
    assert list(again.training) == list(first.training)


def test_a_block_left_in_the_workspace_by_the_publisher_is_read_without_the_store(
    tmp_path: Path,
) -> None:
    # The Catalog's archive keeps the block it wrote under the same name in the same workspace,
    # so a machine that just published reads its corpus back without fetching it.
    store = InMemoryArtifactStore()
    workspace = tmp_path / "shared"
    published = publish(store, workspace / "scratch")
    store.get_file(published.block, workspace / published.block.checksum.digest)

    read = BlockTrainingCorpusReader(_refusing(store), workspace).read(published.manifest)

    assert read.shape == published.corpus.shape


def test_a_block_in_the_workspace_that_does_not_hash_to_its_name_is_fetched_again(
    tmp_path: Path,
) -> None:
    # The workspace outlives the process and is written by another one: what sits under a
    # digest is hashed before a run is signed over it, and replaced where it does not match.
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    workspace = tmp_path / "reader"
    reader = BlockTrainingCorpusReader(store, workspace)
    expected = list(reader.read(published.manifest).training)
    cached = workspace / published.block.checksum.digest
    cached.write_bytes(cached.read_bytes()[:-1] + b"\x00")

    read = reader.read(published.manifest)

    assert list(read.training) == expected
    assert published.block.checksum.matches(cached.read_bytes())


def test_describing_never_fetches_the_block(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")

    described = BlockTrainingCorpusReader(_refusing(store), tmp_path / "reader").describe(
        published.manifest
    )

    assert described == published.described


def test_bytes_that_are_not_a_manifest_are_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    ref = store.put(b'{"format": "something else"}', Retention.DURABLE)
    reader = BlockTrainingCorpusReader(store, tmp_path)

    with pytest.raises(UnreadablePublishedCorpusError, match="not a published manifest"):
        reader.describe(ref)
    with pytest.raises(UnreadablePublishedCorpusError, match="not a published manifest"):
        reader.read(ref)


def test_a_manifest_whose_block_is_not_a_block_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    block = store.put(b"a block, honestly", Retention.DURABLE)
    manifest = store.put(PublishedCorpusManifestJson().encode(manifest_of(block)))

    with pytest.raises(UnreadablePublishedCorpusError, match="not one this reads"):
        BlockTrainingCorpusReader(store, tmp_path).read(manifest)


def test_a_manifest_whose_block_is_missing_is_reported(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    elsewhere = InMemoryArtifactStore()
    manifest = elsewhere.put(store.get(published.manifest))

    with pytest.raises(ArtifactNotFoundError):
        BlockTrainingCorpusReader(elsewhere, tmp_path / "reader").read(manifest)


def test_a_split_that_leaves_a_side_without_windows_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    # Every unit with windows on the training side: the corpus would score on nothing.
    lopsided = manifest_of(
        published.block,
        training_units=(TRAINING_UNIT, VALIDATION_UNIT),
        validation_units=(EMPTY_UNIT,),
    )
    manifest = store.put(PublishedCorpusManifestJson().encode(lopsided))

    with pytest.raises(InvalidTrainingCorpusError, match="validation side"):
        BlockTrainingCorpusReader(store, tmp_path / "reader").read(manifest)


def _refusing(store: InMemoryArtifactStore) -> RefusingStore:
    refusing = RefusingStore()
    refusing._content = store._content
    return refusing
