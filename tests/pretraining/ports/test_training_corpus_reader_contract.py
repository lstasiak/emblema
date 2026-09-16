"""Contract of the TrainingCorpusReader port, run against every adapter.

Every adapter is handed the same published corpus — a block beside a manifest for the one that
reads the store, the description and the windows outright for the one in memory — and must say
the same about it: what the corpus is, from the manifest alone, and which windows are which side
of the split, at the precision the block stores them.
"""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)
from emblema.pretraining.adapters.in_memory.training_corpus_reader import (
    InMemoryTrainingCorpusReader,
)
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.support.published import PublishedCorpus, publish


class Harness(NamedTuple):
    """A reader under test and the corpus it was given, with what it must give back."""

    reader: TrainingCorpusReader
    published: PublishedCorpus


def block(tmp_path: Path) -> Harness:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    return Harness(BlockTrainingCorpusReader(store, tmp_path / "reader"), published)


def in_memory(tmp_path: Path) -> Harness:
    published = publish(InMemoryArtifactStore(), tmp_path / "publisher")
    reader = InMemoryTrainingCorpusReader()
    reader.publish(published.manifest, published.described, published.corpus)
    return Harness(reader, published)


ADAPTERS: dict[str, Callable[[Path], Harness]] = {"block": block, "in_memory": in_memory}
UNKNOWN = ArtifactRef("durable/nowhere", Checksum.of_bytes(b"nothing under this"))


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Harness:
    build: Callable[[Path], Harness] = request.param
    return build(tmp_path)


def test_the_corpus_is_described_as_the_manifest_states_it(harness: Harness) -> None:
    described = harness.reader.describe(harness.published.manifest)

    assert described == harness.published.described


def test_the_windows_come_back_by_side_of_the_split_at_the_stored_precision(
    harness: Harness,
) -> None:
    expected = harness.published.corpus

    read = harness.reader.read(harness.published.manifest)

    assert list(read.training) == list(expected.training)
    assert list(read.validation) == list(expected.validation)


def test_the_corpus_read_is_named_and_signed_as_the_block_and_the_vocabulary(
    harness: Harness,
) -> None:
    expected = harness.published.corpus

    read = harness.reader.read(harness.published.manifest)

    assert read.shape == expected.shape
    assert read.checksum == harness.published.block.checksum


def test_a_manifest_that_is_not_there_is_reported(harness: Harness) -> None:
    with pytest.raises(ArtifactNotFoundError):
        harness.reader.describe(UNKNOWN)
    with pytest.raises(ArtifactNotFoundError):
        harness.reader.read(UNKNOWN)
