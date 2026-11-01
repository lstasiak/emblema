"""What every detection run is given before anything is fitted: windows, not labels."""

from datetime import UTC, datetime

import pytest

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_detection_windows import (
    DrawDetectionWindows,
    DrawDetectionWindowsCommand,
)
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.contracts.events import FrozenTestSplitOpened
from emblema.evaluation.domain.detection.detection_windows import DetectionWindows
from emblema.evaluation.domain.exceptions import (
    InvalidTaughtEventsError,
    ProtocolMismatchError,
    TaskNotFoundError,
    UnknownGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import MANIFEST, TASK, sides, task, units

NOW = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))
ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0],
    UnitKey("b"): [60.0, 120.0],
    UnitKey("c"): [60.0, 120.0, 180.0],
    UnitKey("d"): [60.0, 120.0],
}
# Unit c holds an event over its last two windows and unit d a gap over its first.
EVENTS = {
    TaskWindow(unit=UnitKey("c"), position=5, ends_at=60.0): 0.0,
    TaskWindow(unit=UnitKey("c"), position=6, ends_at=120.0): 1.0,
    TaskWindow(unit=UnitKey("c"), position=7, ends_at=180.0): 1.0,
    TaskWindow(unit=UnitKey("d"), position=8, ends_at=60.0): AnomalyScheme.EXCLUDED,
    TaskWindow(unit=UnitKey("d"), position=9, ends_at=120.0): 2.0,
}
# The fitting side's own events, where a detector is taught some: unit a holds event 3 over two
# windows, unit b event 4 over one and a gap over the other.
FITTING_EVENTS = EVENTS | {
    TaskWindow(unit=UnitKey("a"), position=0, ends_at=60.0): 0.0,
    TaskWindow(unit=UnitKey("a"), position=1, ends_at=120.0): 3.0,
    TaskWindow(unit=UnitKey("a"), position=2, ends_at=180.0): 3.0,
    TaskWindow(unit=UnitKey("b"), position=3, ends_at=60.0): 4.0,
    TaskWindow(unit=UnitKey("b"), position=4, ends_at=120.0): AnomalyScheme.EXCLUDED,
}
FROZEN = FrozenTestSplit(units=units("d"), source="telemetry/test")
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))


def drawn(
    *,
    save_task: bool = True,
    detecting: bool = True,
    purpose: RunPurpose = RunPurpose.TUNING,
    heard: list[FrozenTestSplitOpened] | None = None,
    truths: dict[TaskWindow, float] = EVENTS,
    events: int = 0,
    seed: int = 1,
) -> DetectionWindows:
    tasks = InMemoryDownstreamTaskRepository()
    if save_task:
        tasks.save(
            task(
                test=FROZEN,
                protocol=EvaluationProtocol.ANOMALY_DETECTION,
                labels=AnomalyScheme(),
                strata=None,
            )
            if detecting
            else task(test=FROZEN)
        )
    subscriptions = InMemoryEventSubscriber()
    if heard is not None:
        subscriptions.subscribe(FrozenTestSplitOpened, heard.append)
    use_case = DrawDetectionWindows(
        tasks,
        InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST),
        InMemoryGroundTruth(by_window=truths),
        OpenTestSplit(
            tasks, SequentialIdGenerator(), FixedClock(NOW), InMemoryEventPublisher(subscriptions)
        ),
    )
    return use_case(
        DrawDetectionWindowsCommand(task=TASK, purpose=purpose, events=events, seed=seed)
    )


def test_the_fitting_side_is_the_tuning_side_whole_and_carries_no_label() -> None:
    windows = drawn()

    assert [(str(w.unit), w.position) for w in windows.fitting] == [
        ("a", 0),
        ("a", 1),
        ("a", 2),
        ("b", 3),
        ("b", 4),
    ]


def test_a_tuning_run_is_answered_on_the_validation_side_with_its_events() -> None:
    heard: list[FrozenTestSplitOpened] = []

    windows = drawn(heard=heard)

    assert [(w.window.position, w.target) for w in windows.scored] == [(5, 0.0), (6, 1.0), (7, 1.0)]
    assert heard == []


def test_the_final_run_is_answered_on_the_frozen_side_and_leaves_out_excluded_windows() -> None:
    heard: list[FrozenTestSplitOpened] = []

    windows = drawn(purpose=RunPurpose.FINAL, heard=heard)

    assert [(w.window.position, w.target) for w in windows.scored] == [(9, 2.0)]
    assert len(heard) == 1


def test_a_selection_run_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="declared before the run"):
        drawn(purpose=RunPurpose.SELECTION)


def test_a_task_that_spends_labels_is_refused() -> None:
    with pytest.raises(ProtocolMismatchError, match="spends labels"):
        drawn(detecting=False)


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(TaskNotFoundError):
        drawn(save_task=False)


def test_a_scored_window_the_ground_truth_says_nothing_about_is_refused() -> None:
    with pytest.raises(UnknownGroundTruthError):
        drawn(truths={})


def test_a_detector_taught_events_is_handed_whole_events_of_the_fitting_side() -> None:
    windows = drawn(truths=FITTING_EVENTS, events=1, seed=1)

    # Under seed 1 the event drawn is the third, both of its windows and nothing of the fourth.
    assert [(w.window.position, w.target) for w in windows.taught] == [(1, 3.0), (2, 3.0)]
    # Every fitting window is still on the fitting side, and the scored side is as before.
    assert len(windows.fitting) == 5
    assert [w.target for w in windows.scored] == [0.0, 1.0, 1.0]


def test_both_events_of_the_side_are_taught_when_two_are_asked_and_a_third_is_refused() -> None:
    both = drawn(truths=FITTING_EVENTS, events=2, seed=1)

    assert sorted({w.target for w in both.taught}) == [3.0, 4.0]
    with pytest.raises(InvalidTaughtEventsError, match="holds 2 events"):
        drawn(truths=FITTING_EVENTS, events=3, seed=1)


def test_a_detector_taught_no_event_labels_no_fitting_window() -> None:
    # The truths say nothing about the fitting windows, and nothing is asked of them.
    assert drawn(events=0).taught == ()
