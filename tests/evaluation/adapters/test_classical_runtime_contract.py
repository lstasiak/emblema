"""One task, one sample, every adapter: what any runtime that fits a classical candidate owes.

The contract says nothing about how good the answers are — a runtime may fit trees or learn the
mean — and everything about what a caller may rely on: exactly the scored windows answered in
the order given, another task's labels taken in when they are offered, a sample of a foreign
task refused, and what was fitted kept only where there is somewhere to keep it. The XGBoost
adapter runs over the published test corpus, whose block holds the four positions the samples
address.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.classical_runtime import InMemoryClassicalRuntime
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    ForeignLabelSampleError,
    InvalidClassicalOutcomeError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.ports.classical_runtime import ClassicalRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.evaluation.support import TASK, labelled, recipe, task

OTHER_TASK = TaskId(UUID(int=11))
# The block holds four windows; three are fitted on and the held-out one is scored.
SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("a", 1, 15.0, 9.0), labelled("b", 3, 10.0, 2.0)),
    budget=LabelBudget.of(3),
    seed=7,
)
SCORED = (labelled("c", 2, 10.0, 8.0),)
AGGREGATED = FeatureScheme.CHANNEL_AGGREGATED


class Adapted(NamedTuple):
    """An adapter, the task it answers for, and where what it fits would be kept."""

    runtime: ClassicalRuntime
    task: DownstreamTask
    store: ArtifactStore


def in_memory(tmp_path: Path, *, keeping: bool) -> Adapted:
    store = InMemoryArtifactStore()
    return Adapted(InMemoryClassicalRuntime(store if keeping else None), task(), store)


def over_xgboost(tmp_path: Path, *, keeping: bool) -> Adapted:
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.xgboost.xgboost_classical_runtime import (
        XgboostClassicalRuntime,
    )
    from tests.support.published import publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = XgboostClassicalRuntime(
        PublishedCorpusBlocks(store, tmp_path / "workspace"),
        threads=1,
        store=store if keeping else None,
    )
    return Adapted(runtime, replace(task(), manifest=manifest), store)


ADAPTERS = {"in memory": in_memory, "xgboost": over_xgboost}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def adapted(request: pytest.FixtureRequest, tmp_path: Path) -> Adapted:
    return request.param(tmp_path, keeping=False)


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def keeping(request: pytest.FixtureRequest, tmp_path: Path) -> Adapted:
    return request.param(tmp_path, keeping=True)


def source_of(adapted: Adapted) -> FittingSource:
    """Another task over the same corpus, whose labels pull every answer towards nothing."""
    elsewhere = replace(adapted.task, task_id=OTHER_TASK)
    at_nothing = tuple(replace(window, target=0.0) for window in SAMPLE.windows)
    return FittingSource(
        task=elsewhere, sample=replace(SAMPLE, task=OTHER_TASK, windows=at_nothing)
    )


@pytest.mark.parametrize("scheme", list(FeatureScheme))
def test_every_scored_window_is_answered_in_the_order_given(
    adapted: Adapted, scheme: FeatureScheme
) -> None:
    outcome = adapted.runtime.fit(recipe(scheme), adapted.task, SAMPLE, (), SCORED, retain=False)

    assert [p.window for p in outcome.predictions] == [w.window for w in SCORED]
    assert [p.target for p in outcome.predictions] == [w.target for w in SCORED]
    assert outcome.seconds >= 0.0


def test_a_fit_answers_with_finite_numbers(adapted: Adapted) -> None:
    outcome = adapted.runtime.fit(recipe(), adapted.task, SAMPLE, (), SCORED, retain=False)

    assert outcome.rmse >= 0.0


def test_the_labels_of_a_source_task_change_what_the_fit_answers(adapted: Adapted) -> None:
    alone = adapted.runtime.fit(recipe(AGGREGATED), adapted.task, SAMPLE, (), SCORED, retain=False)

    with_source = adapted.runtime.fit(
        recipe(AGGREGATED, sources=(OTHER_TASK,)),
        adapted.task,
        SAMPLE,
        (source_of(adapted),),
        SCORED,
        retain=False,
    )

    assert with_source.predictions[0].predicted < alone.predictions[0].predicted


def test_a_sample_of_another_task_is_refused(adapted: Adapted) -> None:
    foreign = replace(SAMPLE, task=OTHER_TASK)

    with pytest.raises(ForeignLabelSampleError):
        adapted.runtime.fit(recipe(), adapted.task, foreign, (), SCORED, retain=False)


def test_a_fit_with_nothing_to_score_is_refused(adapted: Adapted) -> None:
    with pytest.raises(InvalidClassicalOutcomeError):
        adapted.runtime.fit(recipe(), adapted.task, SAMPLE, (), (), retain=False)


def test_a_fit_asked_to_keep_what_it_produced_names_bytes_that_exist(keeping: Adapted) -> None:
    outcome = keeping.runtime.fit(recipe(), keeping.task, SAMPLE, (), SCORED, retain=True)

    assert outcome.artifact is not None
    assert keeping.store.get(outcome.artifact)


def test_a_runtime_with_nowhere_to_keep_what_it_fits_refuses_to_keep_it(adapted: Adapted) -> None:
    with pytest.raises(CandidateNotRetainableError):
        adapted.runtime.fit(recipe(), adapted.task, SAMPLE, (), SCORED, retain=True)
