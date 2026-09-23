"""A task defined and drawn over a corpus this test really publishes.

Everything in between is the code that runs in earnest: the turbofan reader, the tokeniser, the
block writer, a store on disk, the published manifest, and the ground truth read from the same
files the corpus was read from. What the unit tests hold apart — a manifest, a placement, a
failure time — has to line up here, where nothing is arranged for it.

The sample holds two engines, so one tunes and one validates; the split of the published corpus
decides which is which, exactly as it does on the full corpus.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.readers.cmapss_ground_truth import CmapssGroundTruth
from emblema.evaluation.application.use_cases.define_downstream_task import (
    DefineDownstreamTask,
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.support.corpora import CORPUS, SAMPLE, SAMPLE_WINDOW, SUBSET, publish_command
from tests.support.settings import unreachable_store

ENGINES = (UnitKey("FD001/39"), UnitKey("FD001/91"))
# Recorded for 128 and 135 cycles, so each fails one cycle past its last.
FAILURES = {ENGINES[0]: 129.0, ENGINES[1]: 136.0}
HELD = FrozenTestSplit(units=frozenset({UnitKey("FD001/test/1")}), source="cmapss/test/FD001")
CEILING = 40.0


class Published(NamedTuple):
    """The published corpus and the task drawn over it."""

    drawing: DrawLabelBudget
    task: TaskId
    store: ArtifactStore
    block: Path


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> Published:
    root = tmp_path_factory.mktemp("published")
    store = LocalDirectoryArtifactStore(root / "store")
    publishing = CompositionRoot(
        unreachable_store(),
        corpus=CORPUS,
        corpus_root=SAMPLE,
        workspace=root / "blocks",
        subsets=(SUBSET,),
        corpora=InMemoryCorpusRepository(),
        store=store,
    )
    manifest = publishing.services.publish_corpus(publish_command())

    workspace = root / "workspace"
    corpus = BlockCorpusWindows(PublishedCorpusBlocks(store, workspace))
    tasks = InMemoryDownstreamTaskRepository()
    task = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        DefineDownstreamTaskCommand(
            manifest=manifest,
            units=frozenset(ENGINES),
            test=HELD,
            protocol=EvaluationProtocol.LABEL_BUDGET,
            labels=RemainingLifeScheme(CEILING),
            strata=TargetBins(2),
        )
    )
    block = PublishedCorpusManifestJson().decode(store.get(manifest)).block
    return Published(
        DrawLabelBudget(tasks, corpus, CmapssGroundTruth(SAMPLE)),
        task,
        store,
        workspace / block.checksum.digest,
    )


def drawn(published: Published, budget: LabelBudget, seed: int = 1) -> LabelSample:
    return published.drawing(DrawLabelBudgetCommand(task=published.task, budget=budget, seed=seed))


def test_the_task_takes_one_engine_to_tune_on_and_one_to_validate_on(
    published: Published,
) -> None:
    sample = drawn(published, LabelBudget.everything())

    tuning = {str(labelled.window.unit) for labelled in sample.windows}
    assert len(tuning) == 1
    assert tuning < {str(engine) for engine in ENGINES}


def test_every_window_is_labelled_from_the_failure_of_its_own_engine(
    published: Published,
) -> None:
    sample = drawn(published, LabelBudget.everything())
    engine = sample.windows[0].window.unit

    for labelled in sample.windows:
        expected = min(FAILURES[engine] - labelled.window.ends_at, CEILING)
        assert labelled.target == pytest.approx(expected)


def test_the_last_window_of_a_run_to_failure_all_but_reaches_it(published: Published) -> None:
    sample = drawn(published, LabelBudget.everything())
    engine = sample.windows[0].window.unit

    nearest = min(labelled.target for labelled in sample.windows)
    # The tail the windowing leaves out is shorter than one stride, so the label nearest failure
    # is under a stride away from zero — the reason this task needs no window anchored at the end.
    assert 0 <= nearest < SAMPLE_WINDOW.stride
    assert max(labelled.window.ends_at for labelled in sample.windows) < FAILURES[engine]


def test_a_budget_drawn_over_the_real_block_repeats(published: Published) -> None:
    first = drawn(published, LabelBudget.of(4))
    again = drawn(published, LabelBudget.of(4))

    assert first == again
    assert len(first.windows) == 4


def test_the_positions_a_sample_carries_address_the_windows_in_the_block(
    published: Published,
) -> None:
    sample = drawn(published, LabelBudget.of(4))
    block = WindowBlock(published.block)

    # Whoever trains reads tokens by these positions, so each has to land on the window the
    # sample says it does, in the block as published.
    for labelled in sample.windows:
        assert block.extent_of(labelled.window.position)[1] == labelled.window.ends_at
        assert len(block[labelled.window.position].channel_ids) > 0
