import pytest

from emblema.evaluation.domain.exceptions import InvalidGradientBoostingSpecError
from tests.evaluation.support import boosting, recipe


@pytest.mark.parametrize(("field", "value"), [("rounds", 0), ("max_depth", 0), ("threads", 0)])
def test_a_fit_that_grows_nothing_or_runs_on_nothing_is_refused(field: str, value: int) -> None:
    with pytest.raises(InvalidGradientBoostingSpecError, match="must be positive"):
        boosting(**{field: value})


@pytest.mark.parametrize("rate", [0.0, -0.1, float("inf"), float("nan")])
def test_a_rate_that_is_not_a_positive_number_is_refused(rate: float) -> None:
    with pytest.raises(InvalidGradientBoostingSpecError, match="learning_rate"):
        boosting(learning_rate=rate)


@pytest.mark.parametrize("field", ["row_share", "feature_share"])
@pytest.mark.parametrize("share", [0.0, 1.1, -0.5, float("nan")])
def test_a_share_outside_the_unit_interval_is_refused(field: str, share: float) -> None:
    with pytest.raises(InvalidGradientBoostingSpecError, match=r"must lie in \(0, 1\]"):
        boosting(**{field: share})


def test_a_share_of_everything_is_allowed() -> None:
    assert boosting(row_share=1.0, feature_share=1.0).row_share == 1.0


@pytest.mark.parametrize("field", ["min_leaf_weight", "l2_penalty"])
@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_a_penalty_that_is_negative_or_not_a_number_is_refused(field: str, value: float) -> None:
    with pytest.raises(InvalidGradientBoostingSpecError, match="finite and not negative"):
        boosting(**{field: value})


def test_no_penalty_at_all_is_allowed() -> None:
    assert boosting(min_leaf_weight=0.0, l2_penalty=0.0).l2_penalty == 0.0


def test_the_threads_a_fit_runs_on_are_part_of_what_it_reports() -> None:
    # Two machines agree on what a recipe produced only if they agree on this, so it travels
    # with the recipe rather than with whichever worker picked the cell up.
    assert recipe(boosting=boosting(threads=4)).parameters()["threads"] == 4
