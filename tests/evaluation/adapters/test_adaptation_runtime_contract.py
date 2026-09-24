"""One task, one sample, every adapter: what any runtime owes the port.

A runtime may learn anything or nothing, so the contract says nothing about how good the answers
are; it holds every adapter to answering exactly the validation windows it was handed, in the
order given, to placing the outcome on the curve by the identities it was given, and to refusing
a sample of another task's labels. The torch adapter runs over the published test corpus, whose
block holds the four positions the sample and the validation side address.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.adaptation_runtime import InMemoryAdaptationRuntime
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    ForeignLabelSampleError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.ports.adaptation_runtime import AdaptationRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.evaluation.support import TASK, labelled, plan, task

SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("b", 3, 10.0, 9.0)),
    budget=LabelBudget.of(2),
    seed=7,
)
VALIDATION = (labelled("c", 2, 10.0, 8.0), labelled("c", 1, 15.0, 3.0))


class Adapted(NamedTuple):
    """An adapter and the task it answers for."""

    runtime: AdaptationRuntime
    task: DownstreamTask


@pytest.fixture(params=["in memory", "torch"])
def adapted(request: pytest.FixtureRequest, tmp_path: Path) -> Adapted:
    if request.param == "in memory":
        return Adapted(InMemoryAdaptationRuntime(), task())
    pytest.importorskip("torch")
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.torch.torch_adaptation_runtime import (
        TorchAdaptationRuntime,
    )
    from tests.support.backbones import SmallBackbones
    from tests.support.published import CHANNELS, publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=len(CHANNELS)),
        PublishedCorpusBlocks(store, tmp_path / "workspace"),
        device="cpu",
    )
    return Adapted(runtime, replace(task(), manifest=manifest))


@pytest.mark.parametrize("mode", list(TransferMode))
def test_every_validation_window_is_answered_in_the_order_given(
    adapted: Adapted, mode: TransferMode
) -> None:
    outcome = adapted.runtime.adapt(plan(mode), adapted.task, SAMPLE, VALIDATION, retain=False)

    assert [p.window for p in outcome.predictions] == [w.window for w in VALIDATION]
    assert [p.target for p in outcome.predictions] == [w.target for w in VALIDATION]


def test_the_outcome_carries_what_places_it_on_the_curve(adapted: Adapted) -> None:
    stated = plan(TransferMode.FROZEN_PROBE)

    outcome = adapted.runtime.adapt(stated, adapted.task, SAMPLE, VALIDATION, retain=False)

    assert (outcome.plan, outcome.task) == (stated, TASK)
    assert (outcome.budget, outcome.sample_seed) == (LabelBudget.of(2), 7)
    assert len(outcome.training_losses) == stated.schedule.epochs_over(len(SAMPLE.windows))
    assert outcome.trainable_parameters >= 1


def test_a_sample_of_another_task_is_refused(adapted: Adapted) -> None:
    foreign = LabelSample(
        task=TaskId(UUID(int=9)), windows=SAMPLE.windows, budget=SAMPLE.budget, seed=SAMPLE.seed
    )

    with pytest.raises(ForeignLabelSampleError):
        adapted.runtime.adapt(plan(), adapted.task, foreign, VALIDATION, retain=False)


def test_a_run_with_nothing_to_score_is_refused(adapted: Adapted) -> None:
    with pytest.raises(InvalidScoredOutcomeError):
        adapted.runtime.adapt(plan(), adapted.task, SAMPLE, (), retain=False)
