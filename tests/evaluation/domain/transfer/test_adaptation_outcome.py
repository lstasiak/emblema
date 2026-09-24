from dataclasses import replace
from typing import Any

import pytest

from emblema.evaluation.domain.exceptions import (
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from tests.evaluation.support import TASK, adaptation_schedule, plan, prediction

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
        labelled_windows=2,
        labelled_units=2,
        trainable_parameters=257,
        training_losses=(0.5, 0.25),
        predictions=predictions,
        seconds=1.5,
        artifact=None,
    )
    return replace(stated, **overrides)


def test_an_adaptation_is_held_to_what_every_scored_outcome_is_held_to() -> None:
    with pytest.raises(InvalidScoredOutcomeError, match="at least one window"):
        outcome(())


def test_the_outcome_counts_the_optimiser_steps_the_run_took() -> None:
    # Two labelled windows in batches of two: one step an epoch, two epochs.
    assert outcome().optimiser_steps == 2
    lifted = plan(schedule=adaptation_schedule(epochs=2, min_steps=3, batch_size=2))
    assert outcome(plan=lifted, training_losses=(0.5, 0.4, 0.3)).optimiser_steps == 3
