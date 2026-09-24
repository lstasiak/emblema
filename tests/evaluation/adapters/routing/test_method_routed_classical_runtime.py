from emblema.evaluation.adapters.in_memory.classical_runtime import InMemoryClassicalRuntime
from emblema.evaluation.adapters.routing.method_routed_classical_runtime import (
    MethodRoutedClassicalRuntime,
)
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from tests.evaluation.support import TASK, convolutions, labelled, recipe, task

SAMPLE = LabelSample(
    task=TASK, windows=(labelled("a", 0, 10.0, 5.0),), budget=LabelBudget.of(1), seed=1
)


def routed(fitted: ClassicalRecipe) -> tuple[int, int]:
    trees, rocket = InMemoryClassicalRuntime(), InMemoryClassicalRuntime()
    runtime = MethodRoutedClassicalRuntime(trees=trees, convolutions=rocket)

    runtime.fit(fitted, task(), SAMPLE, (), (labelled("c", 2, 10.0, 8.0),), retain=False)

    return len(trees.fittings), len(rocket.fittings)


def test_a_recipe_of_trees_goes_to_the_runtime_that_grows_them() -> None:
    assert routed(recipe()) == (1, 0)


def test_a_recipe_of_convolutions_goes_to_the_runtime_that_fits_them() -> None:
    assert routed(recipe(method=convolutions())) == (0, 1)
