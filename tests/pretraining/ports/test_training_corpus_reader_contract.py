"""Contract of the TrainingCorpusReader port, run against every adapter.

Every adapter is handed the same published corpus — a block beside a manifest for the one that
reads the store, the description and the windows outright for the one in memory — and must say
the same about it: what the corpus is, from the manifest alone, which windows are which side of
the split, at the precision the block stores them, and which of the training units a share of
the corpus reads.
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
from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.ports.training_corpus_reader import TrainingCorpusReader
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.support.experiments import share
from tests.support.published import (
    EMPTY_UNIT,
    OTHER_TRAINING_UNIT,
    TRAINING_UNIT,
    TRAINING_UNITS,
    PublishedCorpus,
    publish,
)

WHOLE = share()


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
    reader.publish(
        published.manifest,
        published.described,
        published.corpus,
        TRAINING_UNITS,
        empty_units=(EMPTY_UNIT,),
    )
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

    read = harness.reader.read(harness.published.manifest, WHOLE)

    assert list(read.training) == list(expected.training)
    assert list(read.validation) == list(expected.validation)


def test_the_corpus_read_is_named_and_signed_as_the_block_and_the_vocabulary(
    harness: Harness,
) -> None:
    expected = harness.published.corpus

    read = harness.reader.read(harness.published.manifest, WHOLE)

    assert read.shape == expected.shape
    assert read.checksum == harness.published.block.checksum


def test_a_manifest_that_is_not_there_is_reported(harness: Harness) -> None:
    with pytest.raises(ArtifactNotFoundError):
        harness.reader.describe(UNKNOWN)
    with pytest.raises(ArtifactNotFoundError):
        harness.reader.read(UNKNOWN, WHOLE)


# Shares and seeds chosen so that each picks a different set of the three training units, one of
# them the unit without windows, and one leaves that unit out altogether.
@pytest.mark.parametrize(("fraction", "seed"), [(1 / 3, 1), (0.5, 4), (0.5, 3)])
def test_a_share_reads_the_units_it_picks_and_the_whole_validation_side(
    harness: Harness, fraction: float, seed: int
) -> None:
    """Which units a share picks is the share's rule; the reader hands it the manifest's."""
    stated = share(fraction, seed=seed)
    chosen = stated.select((TRAINING_UNIT, EMPTY_UNIT, OTHER_TRAINING_UNIT))
    expected = harness.published.corpus

    read = harness.reader.read(harness.published.manifest, stated)

    assert list(read.training) == [
        window
        for window, unit in zip(expected.training, TRAINING_UNITS, strict=True)
        if unit in chosen
    ]
    assert list(read.validation) == list(expected.validation)
    assert read.checksum == expected.checksum


def test_a_share_that_picks_only_the_unit_without_windows_leaves_no_training_side(
    harness: Harness,
) -> None:
    # The empty unit is a unit of the corpus: a share small enough to pick it alone reads
    # nothing, which is refused where every corpus is — at its shape.
    lone = share(1 / 3, seed=2)
    assert lone.select((TRAINING_UNIT, EMPTY_UNIT, OTHER_TRAINING_UNIT)) == (EMPTY_UNIT,)

    with pytest.raises(InvalidTrainingCorpusError, match="training side"):
        harness.reader.read(harness.published.manifest, lone)


def test_the_in_memory_reader_refuses_units_that_do_not_match_its_windows(
    tmp_path: Path,
) -> None:
    # Only the in-memory adapter is told its units; naming a different number of them than the
    # corpus has training windows would make a share read a share of something else.
    published = publish(InMemoryArtifactStore(), tmp_path / "publisher")

    with pytest.raises(ValueError, match="units named for"):
        InMemoryTrainingCorpusReader().publish(
            published.manifest,
            published.described,
            published.corpus,
            TRAINING_UNITS[:-1],
        )
