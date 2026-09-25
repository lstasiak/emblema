"""What the use case adds over the labels it is handed: the patch model learns and answers them.

Which windows those are, and which side they come from, is the drawing's business and is held
to account beside it; here the question is only that what was drawn is what the model saw.
"""

from datetime import UTC, datetime

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.ground_truth import InMemoryGroundTruth
from emblema.evaluation.adapters.in_memory.patch_runtime import InMemoryPatchRuntime
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget
from emblema.evaluation.application.use_cases.draw_run_labels import DrawRunLabels
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_patch_training import (
    RunPatchTraining,
    RunPatchTrainingCommand,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.evaluation.support import MANIFEST, TASK, patch_plan, sides, task, units

NOW = UtcDateTime(datetime(2026, 1, 1, tzinfo=UTC))
ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
    UnitKey("d"): [60.0, 120.0],
}
FAILURES = {UnitKey("a"): 300.0, UnitKey("b"): 260.0, UnitKey("c"): 200.0, UnitKey("d"): 210.0}
FROZEN = FrozenTestSplit(units=units("d"), source="turbofans/test")
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))


def run(runtime: InMemoryPatchRuntime, purpose: RunPurpose = RunPurpose.TUNING) -> ScoredOutcome:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task(test=FROZEN))
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)
    lifetimes = InMemoryGroundTruth(FAILURES)
    use_case = RunPatchTraining(
        DrawRunLabels(
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
        ),
        runtime,
    )
    return use_case(
        RunPatchTrainingCommand(
            task=TASK,
            plan=patch_plan(seed=5),
            budget=LabelBudget.of(4),
            sample_seed=3,
            purpose=purpose,
        )
    )


def test_the_model_learns_from_the_budget_and_answers_every_validation_window() -> None:
    runtime = InMemoryPatchRuntime()

    outcome = run(runtime)

    (asked,) = runtime.trainings
    assert len(asked.sample.windows) == 4
    assert asked.sample.seed == 3
    assert asked.plan == patch_plan(seed=5)
    assert {str(w.window.unit) for w in asked.sample.windows} <= {"a", "b"}
    assert [(str(p.window.unit), p.window.position) for p in outcome.predictions] == [
        ("c", 8),
        ("c", 9),
    ]


def test_the_purpose_of_the_run_reaches_the_side_it_is_answered_on() -> None:
    outcome = run(InMemoryPatchRuntime(), purpose=RunPurpose.FINAL)

    assert {prediction.window.unit for prediction in outcome.predictions} == units("d")
