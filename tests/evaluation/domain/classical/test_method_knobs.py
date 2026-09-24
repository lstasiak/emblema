import pytest

from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.exceptions import UnknownKnobError
from tests.evaluation.support import boosting, convolutions

TREES = BoostedTrees(features=FeatureScheme.PER_CHANNEL, boosting=boosting())


def test_a_knob_of_the_trees_is_turned_and_nothing_else_moves() -> None:
    deeper = TREES.tuned("max_depth", "5")

    assert deeper.boosting.max_depth == 5
    assert {k for k, v in deeper.parameters().items() if TREES.parameters()[k] != v} == {
        "max_depth"
    }


def test_the_grid_of_the_convolutions_is_a_knob_under_the_name_the_recipe_records() -> None:
    finer = convolutions().tuned("grid_resolution", "1.28")

    assert finer.parameters()["grid_resolution"] == 1.28


@pytest.mark.parametrize(
    ("knob", "value", "complaint"),
    [
        ("threads", "8", "no knob"),
        ("max_depth", "2.5", "takes a int"),
        ("max_depth", "0", "cannot be 0"),
        ("row_share", "1.5", "cannot be 1.5"),
    ],
)
def test_a_knob_the_trees_lack_or_a_value_they_refuse_is_refused(
    knob: str, value: str, complaint: str
) -> None:
    with pytest.raises(UnknownKnobError, match=complaint):
        TREES.tuned(knob, value)


def test_the_ridge_penalty_is_not_a_knob_of_the_convolutions() -> None:
    with pytest.raises(UnknownKnobError, match="no knob"):
        convolutions().tuned("ridge_penalties", "1")
