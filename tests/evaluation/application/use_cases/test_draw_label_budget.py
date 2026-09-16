from dataclasses import replace

import pytest

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.in_memory.unit_lifetimes import InMemoryUnitLifetimes
from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.domain.exceptions import (
    TaskNotFoundError,
    UnknownUnitLifetimeError,
    UnlabelledWindowError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.target_bins import TargetBins
from tests.evaluation.support import MANIFEST, TASK, sides, task, units

ENDS = {
    UnitKey("a"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("b"): [60.0, 120.0, 180.0, 240.0],
    UnitKey("c"): [60.0, 120.0],
}
FAILURES = {UnitKey("a"): 300.0, UnitKey("b"): 260.0, UnitKey("c"): 200.0}
PUBLISHED = sides(training=units("a", "b"), validation=units("c"))
EVERYTHING = LabelBudget.everything()


def draw(
    budget: LabelBudget = EVERYTHING,
    failures: dict[UnitKey, float] = FAILURES,
    seed: int = 3,
) -> LabelSample:
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(task())
    use_case = DrawLabelBudget(
        tasks,
        InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST),
        InMemoryUnitLifetimes(failures),
    )
    return use_case(DrawLabelBudgetCommand(task=TASK, budget=budget, seed=seed))


def test_the_strata_come_from_the_task_and_not_from_the_request() -> None:
    # Two draws of one budget differ only by how the task was defined; nothing at the point of
    # asking can change how the target is stratified.
    tasks = InMemoryDownstreamTaskRepository()
    tasks.save(replace(task(), strata=TargetBins(1)))
    corpus = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)
    lifetimes = InMemoryUnitLifetimes(FAILURES)
    command = DrawLabelBudgetCommand(task=TASK, budget=LabelBudget.of(4), seed=3)

    coarse = DrawLabelBudget(tasks, corpus, lifetimes)(command)
    tasks.save(replace(task(), strata=TargetBins(4)))
    fine = DrawLabelBudget(tasks, corpus, lifetimes)(command)

    assert coarse != fine


def test_a_window_is_labelled_with_what_was_left_of_its_units_life() -> None:
    sample = draw()

    labelled = {(str(w.window.unit), w.window.ends_at): w.target for w in sample.windows}
    assert labelled[("a", 240.0)] == 60.0
    assert labelled[("b", 240.0)] == 20.0


def test_a_window_early_in_a_life_is_labelled_at_the_ceiling() -> None:
    sample = draw()

    labelled = {(str(w.window.unit), w.window.ends_at): w.target for w in sample.windows}
    assert labelled[("a", 60.0)] == 125.0


def test_only_the_tuning_side_is_drawn_from() -> None:
    sample = draw()

    assert {str(labelled.window.unit) for labelled in sample.windows} == {"a", "b"}


def test_the_same_budget_and_seed_draw_the_same_windows() -> None:
    assert draw(LabelBudget.of(4)) == draw(LabelBudget.of(4))


def test_another_seed_draws_other_windows() -> None:
    assert draw(LabelBudget.of(4), seed=3) != draw(LabelBudget.of(4), seed=9)


def test_a_tuning_unit_without_a_known_failure_stops_the_draw() -> None:
    with pytest.raises(UnknownUnitLifetimeError, match="no failure time"):
        draw(failures={UnitKey("a"): 300.0, UnitKey("c"): 200.0})


def test_a_unit_that_failed_before_its_last_window_ended_stops_the_draw() -> None:
    with pytest.raises(UnlabelledWindowError, match="reaches past the failure"):
        draw(failures={**FAILURES, UnitKey("b"): 200.0})


def test_drawing_from_a_task_nobody_defined_is_refused() -> None:
    use_case = DrawLabelBudget(
        InMemoryDownstreamTaskRepository(),
        InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST),
        InMemoryUnitLifetimes(FAILURES),
    )

    with pytest.raises(TaskNotFoundError):
        use_case(DrawLabelBudgetCommand(task=TASK, budget=LabelBudget.of(2), seed=3))
