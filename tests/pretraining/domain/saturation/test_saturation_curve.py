from collections.abc import Callable

import pytest

from emblema.pretraining.domain.exceptions import InvalidSaturationCurveError
from emblema.pretraining.domain.saturation.saturation_curve import STEP_TOLERANCE, SaturationCurve
from emblema.pretraining.domain.saturation.saturation_point import SaturationPoint


def point(
    fraction: float,
    validation: float,
    *,
    training: float | None = None,
    steps: int = 1000,
    references: tuple[float, float] = (1.0, 1.0),
) -> SaturationPoint:
    return SaturationPoint(
        fraction=fraction,
        training_loss=validation if training is None else training,
        validation_loss=validation,
        training_reference=references[0],
        validation_reference=references[1],
        steps=steps,
    )


def test_the_gains_are_the_relative_fall_of_validation_from_share_to_share() -> None:
    curve = SaturationCurve((point(0.25, 0.4), point(0.5, 0.3), point(1.0, 0.27)))

    assert curve.gains() == pytest.approx((0.25, 0.1))
    assert curve.largest == point(1.0, 0.27)


def test_a_loss_that_rises_is_a_negative_gain() -> None:
    curve = SaturationCurve((point(0.5, 0.3), point(1.0, 0.33)))

    assert curve.gains() == pytest.approx((-0.1,))


def test_a_point_whose_validation_is_zero_gains_nothing_rather_than_dividing_by_it() -> None:
    curve = SaturationCurve((point(0.5, 0.0), point(1.0, 0.0)))

    assert curve.gains() == (0.0,)


def test_the_ratio_of_validation_to_training_is_read_off_a_point() -> None:
    assert point(1.0, 0.3, training=0.2).generalisation_ratio == pytest.approx(1.5)
    assert point(1.0, 0.3, training=0.0).generalisation_ratio == float("inf")


def test_each_side_is_read_against_its_own_trivial_predictor() -> None:
    # A validation side five times harder for every predictor: the same relative loss on both
    # sides is no gap at all, and the relative losses are what the figure draws.
    harder = point(1.0, 1.5, training=0.3, references=(1.0, 5.0))

    assert harder.relative_validation_loss == pytest.approx(0.3)
    assert harder.relative_training_loss == pytest.approx(0.3)
    assert harder.generalisation_ratio == pytest.approx(1.0)


def test_points_within_the_step_tolerance_are_one_curve_and_beyond_it_are_not() -> None:
    within = round(1000 * (1 - STEP_TOLERANCE))
    beyond = within - 1

    SaturationCurve((point(0.5, 0.3, steps=within), point(1.0, 0.2)))
    with pytest.raises(InvalidSaturationCurveError, match="further apart"):
        SaturationCurve((point(0.5, 0.3, steps=beyond), point(1.0, 0.2)))


def test_a_curve_of_one_share_is_refused() -> None:
    with pytest.raises(InvalidSaturationCurveError, match="two shares"):
        SaturationCurve((point(1.0, 0.2),))


@pytest.mark.parametrize("fractions", [(0.5, 0.25), (0.5, 0.5)])
def test_shares_must_increase(fractions: tuple[float, float]) -> None:
    with pytest.raises(InvalidSaturationCurveError, match="increase"):
        SaturationCurve(tuple(point(fraction, 0.2) for fraction in fractions))


# Each point is built through the typed helper, one field out of range, so that the checkers see
# the fields a point has rather than a mapping of anything.
@pytest.mark.parametrize(
    ("field", "build"),
    [
        ("fraction", lambda: point(0.0, 0.2)),
        ("fraction", lambda: point(1.5, 0.2)),
        ("training_loss", lambda: point(1.0, 0.2, training=-0.1)),
        ("validation_loss", lambda: point(1.0, float("nan"), training=0.2)),
        ("training_reference", lambda: point(1.0, 0.2, references=(0.0, 1.0))),
        ("validation_reference", lambda: point(1.0, 0.2, references=(1.0, float("inf")))),
        ("steps", lambda: point(1.0, 0.2, steps=0)),
    ],
)
def test_a_point_no_run_could_have_measured_is_refused(
    field: str, build: Callable[[], SaturationPoint]
) -> None:
    with pytest.raises(InvalidSaturationCurveError, match=field):
        build()
