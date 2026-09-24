"""The names the classical baselines compete under, and which of them may span corpora.

A stored campaign refers to its candidates by these names, so they are asserted here rather
than left to whichever process happened to build the register last.
"""

from uuid import UUID

from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from tests.evaluation.support import boosting, convolutions

ELSEWHERE = TaskId(UUID(int=31))


def over(*sources: TaskId) -> dict[str, ClassicalArm]:
    return {str(arm.ref): arm for arm in KnownBaselines.over(boosting(), convolutions(), sources)}


def test_every_baseline_is_registered_under_the_name_it_is_competed_by() -> None:
    assert list(over()) == [str(ref) for ref in KnownBaselines.refs()]


def test_only_the_layout_independent_baseline_takes_in_the_tasks_the_process_serves() -> None:
    drawing = {name for name, arm in over(ELSEWHERE).items() if arm.sources}

    assert drawing == {str(KnownBaselines.ACROSS_CHANNELS)}


def test_every_baseline_stands_up_as_a_recipe_with_the_sources_it_was_given() -> None:
    # The recipe is what refuses a method that cannot span layouts, so building one from each
    # arm is the proof that its method and its sources belong together.
    for arm in over(ELSEWHERE).values():
        ClassicalRecipe(method=arm.method, seed=1, sources=arm.sources)


def test_a_process_told_of_no_other_task_still_supplies_every_baseline() -> None:
    assert [arm.sources for arm in over().values()] == [()] * len(KnownBaselines.refs())
