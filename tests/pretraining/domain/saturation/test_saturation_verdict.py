"""Each verdict is earned by a curve that should earn it and refused to one that should not."""

import pytest

from emblema.pretraining.domain.saturation.saturation_curve import SaturationCurve
from emblema.pretraining.domain.saturation.saturation_point import SaturationPoint
from emblema.pretraining.domain.saturation.saturation_verdict import (
    GENERALISATION_GAP,
    NOT_LEARNT_FROM,
    PLATEAU_GAIN,
    SaturationVerdict,
    judge,
)


def curve(*validation: float, training_at_largest: float | None = None) -> SaturationCurve:
    """A curve over doublings ending at the whole corpus, training equal to validation unless
    said otherwise at the largest share."""
    fractions = [1.0 / 2**index for index in reversed(range(len(validation)))]
    points = []
    for fraction, loss in zip(fractions, validation, strict=True):
        training = loss
        if fraction == 1.0 and training_at_largest is not None:
            training = training_at_largest
        points.append(
            SaturationPoint(
                fraction=fraction,
                training_loss=training,
                validation_loss=loss,
                training_reference=1.0,
                validation_reference=1.0,
                steps=100,
            )
        )
    return SaturationCurve(tuple(points))


def test_a_loss_still_falling_at_the_whole_corpus_is_data_limited() -> None:
    judgement = judge(curve(0.5, 0.4, 0.3))

    assert judgement.verdict is SaturationVerdict.DATA_LIMITED
    assert judgement.gain == pytest.approx(0.25)
    assert "more of it would help" in judgement.reading


def test_a_last_step_that_lowers_the_loss_by_less_than_the_plateau_gain_is_saturated() -> None:
    plateau = curve(0.5, 0.3, 0.3 * (1 - PLATEAU_GAIN) + 1e-6)
    falling = curve(0.5, 0.3, 0.3 * (1 - PLATEAU_GAIN) - 1e-6)

    assert judge(plateau).verdict is SaturationVerdict.SATURATED
    assert judge(falling).verdict is SaturationVerdict.DATA_LIMITED


def test_a_loss_that_rises_with_more_data_is_saturated_too() -> None:
    assert judge(curve(0.5, 0.3, 0.32)).verdict is SaturationVerdict.SATURATED


def test_validation_far_above_training_at_the_whole_corpus_is_overfitting_before_anything() -> None:
    # The loss falls steeply share to share, which alone would read as data-limited.
    wide = curve(0.5, 0.4, 0.3, training_at_largest=0.3 / GENERALISATION_GAP - 1e-6)
    narrow = curve(0.5, 0.4, 0.3, training_at_largest=0.3 / GENERALISATION_GAP + 1e-6)

    assert judge(wide).verdict is SaturationVerdict.OVERFITTING
    assert judge(wide).generalisation_ratio > GENERALISATION_GAP
    assert judge(narrow).verdict is SaturationVerdict.DATA_LIMITED


def test_a_run_no_better_than_the_channel_mean_on_the_held_out_side_learnt_nothing_first() -> None:
    # Training far under validation, which alone would read as overfitting; more data lowering
    # the loss, which alone would read as data-limited.
    nothing = curve(1.2, 1.1, NOT_LEARNT_FROM, training_at_largest=0.1)
    something = curve(1.2, 1.1, NOT_LEARNT_FROM - 1e-6, training_at_largest=0.1)

    assert judge(nothing).verdict is SaturationVerdict.NOT_LEARNT
    assert "nothing carried over" in judge(nothing).reading
    assert judge(something).verdict is SaturationVerdict.OVERFITTING


def test_the_judgement_names_the_step_it_read() -> None:
    assert "from 50% to 100% of the training units" in judge(curve(0.5, 0.4, 0.3)).reading
