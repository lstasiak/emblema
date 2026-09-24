"""A candidate with no weights, run over the same labels the networks are run over.

The source tasks are what makes one of these capable of transfer, so what is held to account
here is that they are drawn at all, drawn from their own side, and drawn under the coordinates
of the cell rather than under a dial of their own.
"""

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.classical_runtime import InMemoryClassicalRuntime
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_classical_fit import (
    RunClassicalFit,
    RunClassicalFitCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.classical_outcome import ClassicalOutcome
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import TaskNotFoundError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import MANIFEST, TASK, recipe, sides, task, units

NOW = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))
ELSEWHERE = TaskId(UUID(int=21))
MISSING = TaskId(UUID(int=22))

ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
    UnitKey("d"): [60.0, 120.0],
    UnitKey("e"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("f"): [60.0, 120.0, 180.0, 240.0],
}
FAILURES = dict.fromkeys(map(UnitKey, "abcdef"), 300.0)
FROZEN = FrozenTestSplit(units=units("d"), source="turbofans/test")
PUBLISHED = sides(training=units("a", "b", "e", "f"), validation=units("c"))
FOUR = LabelBudget.of(4)
ACROSS = recipe(FeatureScheme.CHANNEL_AGGREGATED)


def elsewhere() -> DownstreamTask:
    """A second task over the same corpus, tuned on units the target never touches."""
    return replace(task(tuning=units("e", "f"), validation=units("c")), task_id=ELSEWHERE)


def run(
    made_of: ClassicalRecipe = ACROSS,
    runtime: InMemoryClassicalRuntime | None = None,
    purpose: RunPurpose = RunPurpose.TUNING,
    retain: bool = False,
) -> ClassicalOutcome:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task(test=FROZEN))
    tasks.save(elsewhere())
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)
    lifetimes = InMemoryGroundTruth(FAILURES)
    budgets = DrawLabelBudget(tasks, corpus, lifetimes)
    use_case = RunClassicalFit(
        tasks,
        DrawRunLabels(
            tasks,
            corpus,
            lifetimes,
            budgets,
            OpenTestSplit(
                tasks,
                SequentialIdGenerator(),
                FixedClock(NOW),
                InMemoryEventPublisher(InMemoryEventSubscriber()),
            ),
        ),
        budgets,
        InMemoryClassicalRuntime(InMemoryArtifactStore()) if runtime is None else runtime,
    )
    return use_case(
        RunClassicalFitCommand(
            task=TASK,
            recipe=made_of,
            budget=FOUR,
            sample_seed=3,
            purpose=purpose,
            retain=retain,
        )
    )


def test_the_candidate_learns_the_targets_budget_and_answers_every_scored_window() -> None:
    runtime = InMemoryClassicalRuntime()

    outcome = run(made_of=recipe(), runtime=runtime)

    (fitted,) = runtime.fittings
    assert len(fitted.sample.windows) == 4
    assert {str(w.window.unit) for w in fitted.sample.windows} <= {"a", "b"}
    assert [(str(p.window.unit), p.window.position) for p in outcome.predictions] == [
        ("c", 8),
        ("c", 9),
    ]


def test_a_recipe_naming_no_source_is_fitted_on_the_target_alone() -> None:
    runtime = InMemoryClassicalRuntime()

    run(made_of=recipe(), runtime=runtime)

    (fitted,) = runtime.fittings
    assert fitted.sources == ()


def test_each_source_lends_labels_drawn_from_its_own_tuning_side() -> None:
    runtime = InMemoryClassicalRuntime()

    run(made_of=replace(ACROSS, sources=(ELSEWHERE,)), runtime=runtime)

    (fitted,) = runtime.fittings
    (lent,) = fitted.sources
    assert lent.task.task_id == ELSEWHERE
    assert {str(w.window.unit) for w in lent.sample.windows} <= {"e", "f"}


def test_a_source_contributes_a_budget_of_the_cells_own_size_and_seed() -> None:
    runtime = InMemoryClassicalRuntime()

    run(made_of=replace(ACROSS, sources=(ELSEWHERE,)), runtime=runtime)

    (fitted,) = runtime.fittings
    (lent,) = fitted.sources
    assert lent.sample.budget == fitted.sample.budget
    assert lent.sample.seed == fitted.sample.seed


def test_the_recipe_reaches_the_runtime_that_fits_it() -> None:
    runtime = InMemoryClassicalRuntime()
    named = replace(ACROSS, sources=(ELSEWHERE,), seed=17)

    run(made_of=named, runtime=runtime)

    (fitted,) = runtime.fittings
    assert fitted.recipe == named


def test_the_purpose_of_the_run_reaches_the_side_it_is_answered_on() -> None:
    outcome = run(made_of=recipe(), purpose=RunPurpose.FINAL)

    assert {prediction.window.unit for prediction in outcome.predictions} == units("d")


def test_a_fit_asked_to_keep_what_it_produced_names_an_artifact() -> None:
    outcome = run(made_of=recipe(), retain=True)

    assert outcome.artifact is not None


def test_a_source_task_nobody_stored_is_refused() -> None:
    with pytest.raises(TaskNotFoundError):
        run(made_of=replace(ACROSS, sources=(MISSING,)))
