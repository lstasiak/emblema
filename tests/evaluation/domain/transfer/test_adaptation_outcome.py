from dataclasses import replace
from math import inf, nan
from typing import Any

import pytest

from emblema.evaluation.domain.exceptions import InvalidAdaptationOutcomeError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction
from tests.evaluation.support import TASK, plan, prediction

# Unit b first, so an order by unit is the method's and not the input's.
PREDICTIONS = (
    prediction("b", 2, 30.0, 30.0),
    prediction("a", 0, 10.0, 13.0),
    prediction("a", 1, 20.0, 16.0),
)


def outcome(
    predictions: tuple[WindowPrediction, ...] = PREDICTIONS, **overrides: Any
) -> AdaptationOutcome:
    stated = AdaptationOutcome(
        plan=plan(),
        task=TASK,
        budget=LabelBudget.of(2),
        sample_seed=3,
        trainable_parameters=257,
        training_losses=(0.5, 0.25),
        predictions=predictions,
        seconds=1.5,
    )
    return replace(stated, **overrides)


def test_the_error_over_every_window_is_the_root_of_the_mean_squared_error() -> None:
    assert outcome().rmse == pytest.approx((25.0 / 3) ** 0.5)


def test_the_error_per_unit_pairs_the_windows_by_the_unit_they_came_from() -> None:
    by_unit = outcome().by_unit()

    assert [(str(e.unit), e.squared_error, e.windows) for e in by_unit] == [
        ("a", 25.0, 2),
        ("b", 0.0, 1),
    ]


def test_an_outcome_that_predicts_nothing_is_refused() -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match="at least one window"):
        outcome(())


def test_a_window_predicted_twice_is_refused() -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match="predicted twice"):
        outcome((prediction("a", 0, 10.0, 13.0), prediction("a", 0, 10.0, 12.0)))


def test_the_training_losses_are_one_per_planned_epoch() -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match=r"1 training losses .* 2 planned"):
        outcome(training_losses=(0.5,))


@pytest.mark.parametrize("loss", [nan, inf, -0.1])
def test_a_training_loss_that_is_not_a_finite_non_negative_number_is_refused(loss: float) -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match="training loss"):
        outcome(training_losses=(0.5, loss))


def test_a_run_that_trained_nothing_is_refused() -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match="trained a parameter"):
        outcome(trainable_parameters=0)


@pytest.mark.parametrize("seconds", [-1.0, nan])
def test_a_time_taken_that_is_not_a_finite_non_negative_number_is_refused(seconds: float) -> None:
    with pytest.raises(InvalidAdaptationOutcomeError, match="seconds"):
        outcome(seconds=seconds)
