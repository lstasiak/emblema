"""One corpus, two adapters, one set of answers.

The corpus is the published one every reader contract uses: three units with windows, one of them
held out by the corpus, and a training unit that yielded none and is still named. What a task asks
of it — how the corpus divides its units, and where the windows of the ones it covers sit — has to
come back the same whether the windows are in a block or in a list.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.published import (
    CORPUS,
    EMPTY_UNIT,
    OTHER_TRAINING_UNIT,
    TRAINING_UNIT,
    VALIDATION_UNIT,
    publish,
)

TRAINING, VALIDATION, EMPTY, OTHER = (
    UnitKey(TRAINING_UNIT),
    UnitKey(VALIDATION_UNIT),
    UnitKey(EMPTY_UNIT),
    UnitKey(OTHER_TRAINING_UNIT),
)
SIDES = CorpusSides(
    corpus=CORPUS,
    training=frozenset({TRAINING, EMPTY, OTHER}),
    validation=frozenset({VALIDATION}),
)
# The block holds the two windows of the training unit first, then the one of the held-out unit,
# then the one of the other training unit.
ENDS = {TRAINING: [10.0, 15.0], VALIDATION: [10.0], OTHER: [10.0]}


class Corpus(NamedTuple):
    """An adapter and the reference that addresses the corpus in it."""

    windows: CorpusWindows
    manifest: ArtifactRef


@pytest.fixture(params=["in memory", "block"])
def corpus(request: pytest.FixtureRequest, tmp_path: Path) -> Corpus:
    if request.param == "in memory":
        manifest = ArtifactRef(key="durable/manifest", checksum=Checksum.of_bytes(b"manifest"))
        return Corpus(InMemoryCorpusWindows(SIDES, ENDS, manifest), manifest)
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path)
    return Corpus(
        BlockCorpusWindows(PublishedCorpusBlocks(store, tmp_path / "workspace")),
        published.manifest,
    )


def test_the_corpus_reports_the_sides_it_drew(corpus: Corpus) -> None:
    windows, manifest = corpus

    assert windows.describe(manifest) == SIDES


def test_the_windows_of_a_unit_come_back_with_their_place_and_their_end(corpus: Corpus) -> None:
    windows, manifest = corpus

    read = windows.windows_of(manifest, {TRAINING})

    assert [(str(w.unit), w.position, w.ends_at) for w in read] == [
        (TRAINING_UNIT, 0, 10.0),
        (TRAINING_UNIT, 1, 15.0),
    ]


def test_windows_of_several_units_come_back_in_the_order_the_corpus_holds_them(
    corpus: Corpus,
) -> None:
    windows, manifest = corpus

    read = windows.windows_of(manifest, {VALIDATION, TRAINING})

    assert [w.position for w in read] == [0, 1, 2]


def test_a_unit_that_yielded_no_window_is_named_and_gives_nothing(corpus: Corpus) -> None:
    windows, manifest = corpus

    assert windows.windows_of(manifest, {EMPTY}) == ()


def test_a_unit_the_corpus_does_not_name_is_refused_rather_than_ignored(corpus: Corpus) -> None:
    windows, manifest = corpus

    with pytest.raises(UnreadableTaskCorpusError, match="does not name units"):
        windows.windows_of(manifest, {UnitKey("ghost")})


def test_asking_for_no_unit_gives_no_window(corpus: Corpus) -> None:
    windows, manifest = corpus

    assert windows.windows_of(manifest, set()) == ()
