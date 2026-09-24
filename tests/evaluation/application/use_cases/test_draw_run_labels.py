"""What every run of every candidate is given before anything is fitted.

The labels always come from the tuning side; the purpose of the run decides which side it will
be answered on, and asking for the frozen one is an event. None of it depends on what the
candidate is made of, which is why it is drawn here and not by each runner in turn.
"""

from datetime import UTC, datetime

import pytest

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import (
    DrawRunLabels,
    DrawRunLabelsCommand,
)
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.contracts.events import FrozenTestSplitOpened
from emblema.evaluation.domain.exceptions import (
    InvalidInnerHoldoutError,
    InvalidLabelBudgetError,
    TaskNotFoundError,
    UnknownGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.run_labels import RunLabels
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import MANIFEST, TASK, sides, task, units

NOW = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))

ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
    UnitKey("d"): [60.0, 120.0],
}
FAILURES = {
    UnitKey("a"): 300.0,
    UnitKey("b"): 260.0,
    UnitKey("c"): 200.0,
    UnitKey("d"): 210.0,
}
FROZEN = FrozenTestSplit(units=units("d"), source="turbofans/test")
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))
FOUR = LabelBudget.of(4)


def drawn(
    budget: LabelBudget = FOUR,
    failures: dict[UnitKey, float] = FAILURES,
    save_task: bool = True,
    purpose: RunPurpose = RunPurpose.TUNING,
    heard: list[FrozenTestSplitOpened] | None = None,
    holdout: InnerHoldout | None = None,
) -> RunLabels:
    tasks = InMemoryDownstreamTaskRepository()
    if save_task:
        tasks.save(task(test=FROZEN))
    subscriptions = InMemoryEventSubscriber()
    if heard is not None:
        subscriptions.subscribe(FrozenTestSplitOpened, heard.append)
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)
    lifetimes = InMemoryGroundTruth(failures)
    use_case = DrawRunLabels(
        tasks,
        corpus,
        lifetimes,
        DrawLabelBudget(tasks, corpus, lifetimes),
        OpenTestSplit(
            tasks, SequentialIdGenerator(), FixedClock(NOW), InMemoryEventPublisher(subscriptions)
        ),
    )
    return use_case(
        DrawRunLabelsCommand(task=TASK, budget=budget, seed=3, purpose=purpose, holdout=holdout)
    )


def test_the_budget_is_drawn_from_the_tuning_side_alone() -> None:
    labels = drawn()

    assert len(labels.sample.windows) == 4
    assert {str(w.window.unit) for w in labels.sample.windows} <= {"a", "b"}


def test_a_tuning_run_is_answered_on_the_validation_side_and_opens_nothing() -> None:
    heard: list[FrozenTestSplitOpened] = []

    labels = drawn(heard=heard)

    assert [(str(w.window.unit), w.window.position) for w in labels.scored] == [("c", 8), ("c", 9)]
    assert heard == []


def test_the_final_run_is_answered_on_the_frozen_side_and_leaves_a_record() -> None:
    # The one path the project's single test run takes: the labels still come from the tuning
    # side, the answers are on the frozen units, and the asking is published.
    heard: list[FrozenTestSplitOpened] = []

    labels = drawn(purpose=RunPurpose.FINAL, heard=heard)

    assert {w.window.unit for w in labels.scored} == units("d")
    assert {str(w.window.unit) for w in labels.sample.windows} <= {"a", "b"}
    assert [(event.unit_count, event.source) for event in heard] == [(1, "turbofans/test")]


def test_the_answered_windows_are_labelled_by_the_tasks_own_scheme() -> None:
    labels = drawn()

    # Unit c fails at 200; its windows end at 60 and 120, so 140 is held at the ceiling of 125.
    assert [w.target for w in labels.scored] == [125.0, 80.0]


def test_the_task_travels_with_the_labels_drawn_from_it() -> None:
    labels = drawn()

    assert labels.task.task_id == TASK
    assert labels.task.manifest == MANIFEST


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(TaskNotFoundError):
        drawn(save_task=False)


def test_an_answered_unit_without_a_failure_time_is_refused() -> None:
    with pytest.raises(UnknownGroundTruthError):
        drawn(failures={UnitKey("a"): 300.0, UnitKey("b"): 260.0})


def test_a_budget_wider_than_the_tuning_side_is_refused() -> None:
    with pytest.raises(InvalidLabelBudgetError):
        drawn(budget=LabelBudget.of(9))


def test_a_selection_run_learns_and_is_answered_inside_the_tuning_side_alone() -> None:
    heard: list[FrozenTestSplitOpened] = []

    labels = drawn(purpose=RunPurpose.SELECTION, holdout=InnerHoldout(one_in=2), heard=heard)

    learnt = {str(w.window.unit) for w in labels.sample.windows}
    answered = {str(w.window.unit) for w in labels.scored}
    assert learnt | answered == {"a", "b"}
    assert not learnt & answered
    assert heard == []


def test_a_selection_run_given_no_division_is_refused() -> None:
    with pytest.raises(InvalidInnerHoldoutError, match="division"):
        drawn(purpose=RunPurpose.SELECTION)


def test_a_tuning_run_given_a_division_is_refused() -> None:
    with pytest.raises(InvalidInnerHoldoutError, match="outside the tuning side"):
        drawn(holdout=InnerHoldout(one_in=2))
