from datetime import UTC, datetime

import pytest

from emblema.evaluation.adapters.in_memory.adaptation_runtime import InMemoryAdaptationRuntime
from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_adaptation import (
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.domain.exceptions import TaskNotFoundError, UnknownGroundTruthError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import MANIFEST, TASK, plan, sides, task, units

NOW = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))

ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
}
FAILURES = {UnitKey("a"): 300.0, UnitKey("b"): 260.0, UnitKey("c"): 200.0}
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))
FOUR = LabelBudget.of(4)


def run(
    budget: LabelBudget = FOUR,
    failures: dict[UnitKey, float] = FAILURES,
    runtime: InMemoryAdaptationRuntime | None = None,
    save_task: bool = True,
) -> AdaptationOutcome:
    tasks = InMemoryDownstreamTaskRepository()
    if save_task:
        tasks.save(task())
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)
    lifetimes = InMemoryGroundTruth(failures)
    use_case = RunAdaptation(
        tasks,
        corpus,
        lifetimes,
        DrawLabelBudget(tasks, corpus, lifetimes),
        OpenTestSplit(
            tasks,
            SequentialIdGenerator(),
            FixedClock(NOW),
            InMemoryEventPublisher(InMemoryEventSubscriber()),
        ),
        InMemoryAdaptationRuntime() if runtime is None else runtime,
    )
    return use_case(
        RunAdaptationCommand(task=TASK, plan=plan(TransferMode.LORA), budget=budget, sample_seed=3)
    )


def test_the_candidate_learns_from_the_budget_and_answers_every_validation_window() -> None:
    runtime = InMemoryAdaptationRuntime()

    outcome = run(runtime=runtime)

    (asked,) = runtime.adaptations
    assert len(asked.sample.windows) == 4
    assert {str(w.window.unit) for w in asked.sample.windows} <= {"a", "b"}
    assert [(str(p.window.unit), p.window.position) for p in outcome.predictions] == [
        ("c", 8),
        ("c", 9),
    ]


def test_the_validation_windows_are_labelled_by_the_tasks_own_scheme() -> None:
    runtime = InMemoryAdaptationRuntime()

    run(runtime=runtime)

    (asked,) = runtime.adaptations
    # Unit c fails at 200; its windows end at 60 and 120, so 140 is held at the ceiling of 125.
    assert [w.target for w in asked.validation] == [125.0, 80.0]


def test_the_outcome_places_itself_on_the_curve() -> None:
    outcome = run()

    assert (outcome.task, outcome.budget, outcome.sample_seed) == (TASK, LabelBudget.of(4), 3)
    assert outcome.plan.mode is TransferMode.LORA


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(TaskNotFoundError):
        run(save_task=False)


def test_a_validation_unit_without_a_failure_time_is_refused() -> None:
    with pytest.raises(UnknownGroundTruthError):
        run(failures={UnitKey("a"): 300.0, UnitKey("b"): 260.0})
