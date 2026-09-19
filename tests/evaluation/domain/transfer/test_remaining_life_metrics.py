from dataclasses import replace
from math import exp, inf, nan
from typing import Any

import pytest

from emblema.evaluation.domain.exceptions import InvalidRemainingLifeMetricsError
from emblema.evaluation.domain.transfer.remaining_life_metrics import RemainingLifeMetrics
from tests.evaluation.support import prediction

CEILING = 125.0
# Two engines, the windows of one out of order, so the last window is found by where it ends
# and not by where it is listed. Targets at the ceiling, below it and at zero; answers early,
# exact and late.
ROWS = (
    prediction("a", 1, 100.0, 90.0),
    prediction("a", 0, 125.0, 125.0),
    prediction("b", 2, 50.0, 61.0),
    prediction("b", 3, 0.0, 0.0),
)


def test_every_reading_is_arithmetic_over_the_same_answers() -> None:
    metrics = RemainingLifeMetrics.of(ROWS, ceiling=CEILING)

    assert metrics.windows == 4
    assert metrics.rmse == pytest.approx((221.0 / 4) ** 0.5)
    assert metrics.rmse_below_ceiling == pytest.approx((221.0 / 3) ** 0.5)
    assert metrics.last_window_rmse == pytest.approx((100.0 / 2) ** 0.5)
    # Exact, within a fifth of 100, exact at zero; 61 is just outside a fifth of 50.
    assert metrics.alpha_lambda_accuracy == pytest.approx(3 / 4)
    assert metrics.asymmetric_score == pytest.approx(
        ((exp(10.0 / 13.0) - 1.0) + (exp(11.0 / 10.0) - 1.0)) / 4
    )


def test_a_late_answer_is_penalised_more_than_an_early_one_of_the_same_size() -> None:
    late = RemainingLifeMetrics.of([prediction("a", 0, 50.0, 60.0)], ceiling=CEILING)
    early = RemainingLifeMetrics.of([prediction("a", 0, 50.0, 40.0)], ceiling=CEILING)

    assert late.asymmetric_score > early.asymmetric_score > 0.0
    assert late.rmse == early.rmse


def test_where_every_label_sits_at_the_ceiling_there_is_no_reading_below_it() -> None:
    metrics = RemainingLifeMetrics.of([prediction("a", 0, 125.0, 120.0)], ceiling=CEILING)

    assert metrics.rmse_below_ceiling is None
    assert metrics.rmse == pytest.approx(5.0)


def test_an_answer_late_beyond_what_the_penalty_can_represent_reads_as_infinite() -> None:
    metrics = RemainingLifeMetrics.of(
        [prediction("a", 0, 0.0, 8000.0), prediction("a", 1, 0.0, 0.0)], ceiling=CEILING
    )

    assert metrics.asymmetric_score == inf
    assert metrics.rmse == pytest.approx(8000.0 / 2**0.5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("windows", 0, "at least one window"),
        ("rmse", -1.0, "rmse must be finite"),
        ("rmse", nan, "rmse must be finite"),
        ("rmse_below_ceiling", inf, "rmse_below_ceiling must be finite"),
        ("last_window_rmse", -0.5, "last_window_rmse must be finite"),
        ("alpha_lambda_accuracy", 1.5, "alpha_lambda_accuracy must lie in"),
        ("asymmetric_score", -1.0, "asymmetric_score must be"),
        ("asymmetric_score", nan, "asymmetric_score must be"),
    ],
)
def test_readings_that_are_not_readings_are_refused(field: str, value: Any, message: str) -> None:
    with pytest.raises(InvalidRemainingLifeMetricsError, match=message):
        replace(RemainingLifeMetrics.of(ROWS, ceiling=CEILING), **{field: value})


def test_readings_over_nothing_or_under_no_ceiling_are_refused() -> None:
    with pytest.raises(InvalidRemainingLifeMetricsError, match="at least one prediction"):
        RemainingLifeMetrics.of([], ceiling=CEILING)
    with pytest.raises(InvalidRemainingLifeMetricsError, match="ceiling must be positive"):
        RemainingLifeMetrics.of(ROWS, ceiling=0.0)
