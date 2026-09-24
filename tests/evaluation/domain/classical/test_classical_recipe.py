from uuid import UUID

import pytest

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import InvalidClassicalRecipeError
from tests.evaluation.support import boosting, recipe

OTHER, THIRD = TaskId(UUID(int=7)), TaskId(UUID(int=8))
AGGREGATED = FeatureScheme.CHANNEL_AGGREGATED


def test_a_candidate_that_learns_the_target_alone_names_no_source() -> None:
    assert recipe().sources == ()


def test_a_scheme_that_spans_layouts_may_be_fitted_over_other_tasks() -> None:
    stated = recipe(AGGREGATED, sources=(OTHER, THIRD))

    assert stated.sources == (OTHER, THIRD)


def test_a_channel_bound_scheme_may_not_be_fitted_over_other_tasks() -> None:
    with pytest.raises(InvalidClassicalRecipeError, match="as wide as one corpus has channels"):
        recipe(FeatureScheme.PER_CHANNEL, sources=(OTHER,))


def test_a_source_task_named_twice_is_refused() -> None:
    with pytest.raises(InvalidClassicalRecipeError, match="named twice"):
        recipe(AGGREGATED, sources=(OTHER, OTHER))


def test_the_flattened_recipe_renders_the_same_columns_whatever_it_transfers_from() -> None:
    columns = {
        tuple(recipe(AGGREGATED, sources=sources).parameters())
        for sources in ((), (OTHER,), (OTHER, THIRD))
    }

    assert len(columns) == 1


def test_the_flattened_recipe_names_every_task_it_was_fitted_over() -> None:
    stated = recipe(AGGREGATED, sources=(OTHER, THIRD)).parameters()

    assert stated["sources"] == f"{OTHER} {THIRD}"
    assert stated["features"] == "channel_aggregated"


def test_the_flattened_recipe_tells_two_recipes_apart_by_what_differs() -> None:
    shallow = recipe().parameters()
    deeper = recipe(boosting=boosting(max_depth=9), seed=2).parameters()

    assert {key for key in deeper if deeper[key] != shallow[key]} == {"max_depth", "fit_seed"}
