import random
from math import exp

import pytest

from emblema.evaluation.domain.exceptions import UncalibratableScoresError
from emblema.evaluation.domain.heads.logistic_calibration import LogisticCalibration


def drawn(slope: float, intercept: float, count: int, seed: int) -> tuple[list[float], list[float]]:
    draws = random.Random(seed)
    scores = [draws.uniform(-3.0, 3.0) for _ in range(count)]
    outcomes = [
        1.0 if draws.random() < 1.0 / (1.0 + exp(-(slope * s + intercept))) else 0.0 for s in scores
    ]
    return scores, outcomes


def test_the_fit_recovers_the_calibration_the_outcomes_were_drawn_under() -> None:
    scores, outcomes = drawn(slope=2.0, intercept=-1.0, count=20_000, seed=1)

    fitted = LogisticCalibration.fitted(scores, outcomes)

    assert fitted.slope == pytest.approx(2.0, abs=0.1)
    assert fitted.intercept == pytest.approx(-1.0, abs=0.1)


def test_scores_that_separate_the_outcomes_perfectly_still_give_a_finite_calibration() -> None:
    fitted = LogisticCalibration.fitted([-2.0, -1.0, 1.0, 2.0], [0.0, 0.0, 1.0, 1.0])

    assert 0.5 < fitted.probability(2.0) < 1.0
    assert 0.0 < fitted.probability(-2.0) < 0.5


def test_a_calibration_keeps_the_order_of_the_scores() -> None:
    scores, outcomes = drawn(slope=1.5, intercept=0.5, count=500, seed=2)
    fitted = LogisticCalibration.fitted(scores, outcomes)

    ranked = sorted(scores)
    probabilities = [fitted.probability(score) for score in ranked]

    assert probabilities == sorted(probabilities)


def test_scores_that_say_nothing_calibrate_to_the_smoothed_prevalence() -> None:
    fitted = LogisticCalibration.fitted([0.3] * 10, [1.0, 1.0] + [0.0] * 8)

    # Every score is the same, so only the intercept moves: to the prevalence of the targets
    # Platt smooths, 2 x 3/4 + 8 x 1/10 over 10.
    assert fitted.probability(0.3) == pytest.approx((2 * 0.75 + 8 * 0.1) / 10, abs=1e-4)


def test_the_identity_leaves_a_log_odds_as_it_is() -> None:
    assert LogisticCalibration.identity().probability(0.0) == 0.5
    assert LogisticCalibration.identity().log_odds(1.7) == 1.7


@pytest.mark.parametrize(
    ("scores", "outcomes", "message"),
    [
        ([0.1, 0.2], [1.0], "cannot be calibrated against"),
        ([0.1, 0.2], [1.0, 1.0], "both outcomes"),
        ([0.1, 0.2], [1.0, 0.5], "zero or one"),
        ([0.1, float("nan")], [1.0, 0.0], "finite"),
    ],
)
def test_scores_that_cannot_be_calibrated_are_refused(
    scores: list[float], outcomes: list[float], message: str
) -> None:
    with pytest.raises(UncalibratableScoresError, match=message):
        LogisticCalibration.fitted(scores, outcomes)
