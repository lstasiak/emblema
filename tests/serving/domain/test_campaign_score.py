import math

import pytest

from emblema.serving.domain.exceptions import InvalidCampaignScoreError
from tests.serving.support import score


@pytest.mark.parametrize(
    "overrides",
    [
        {"metric": ""},
        {"metric": " rmse"},
        {"budget": 0},
        {"value": math.nan},
        {"value": math.inf},
        {"repeats": 0},
    ],
    ids=["blank", "padded", "no-window", "nan", "infinite", "no-repeat"],
)
def test_a_score_that_says_nothing_measurable_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidCampaignScoreError):
        score(**overrides)


def test_a_score_over_every_window_of_the_tuning_side_counts_none() -> None:
    assert score(budget=None).budget is None
