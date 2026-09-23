"""The names the classical baselines compete under, and which of them may span corpora.

A stored campaign refers to its candidates by these names, so they are asserted here rather
than left to whichever process happened to build the register last.
"""

from uuid import UUID

from emblema.entrypoints.workers.known_baselines import KnownBaselines
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from tests.evaluation.support import boosting

ELSEWHERE = TaskId(UUID(int=31))


def test_every_baseline_is_registered_under_the_name_it_is_competed_by() -> None:
    assert [arm.ref for arm in KnownBaselines.over()] == list(KnownBaselines.refs())


def test_the_baseline_bound_to_one_layout_draws_on_no_other_task() -> None:
    per_channel, _ = KnownBaselines.over((ELSEWHERE,))

    assert per_channel.ref == KnownBaselines.PER_CHANNEL
    assert per_channel.sources == ()


def test_the_layout_independent_baseline_takes_in_the_tasks_the_process_serves() -> None:
    _, across = KnownBaselines.over((ELSEWHERE,))

    assert across.sources == (ELSEWHERE,)
    # The recipe is what refuses a scheme that cannot span layouts, so building one is the proof
    # that this arm's features and its sources belong together.
    assert ClassicalRecipe(
        features=across.features, boosting=boosting(), seed=1, sources=across.sources
    ).sources == (ELSEWHERE,)


def test_a_process_told_of_no_other_task_still_supplies_both_baselines() -> None:
    assert [arm.sources for arm in KnownBaselines.over()] == [(), ()]
