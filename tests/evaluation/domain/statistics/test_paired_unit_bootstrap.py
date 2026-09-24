"""The procedure on data with a known answer: a shift is found, no shift is not."""

import random

import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedUnitBootstrapError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors

UNITS = 18
WINDOWS = 30


def synthetic(control_rmse: float, candidate_rmse: float, *, seed: int) -> PairedUnitErrors:
    """Units whose per-window errors scatter around two levels, the candidate's the second."""
    draws = random.Random(seed)
    control, candidate = [], []
    for index in range(UNITS):
        unit = UnitKey(f"engine/{index}")
        # Each unit's error level varies, so that the difference between the arms has a spread
        # per unit and the interval has something to be wide about.
        scale = draws.uniform(0.7, 1.3)
        control.append(
            UnitError(unit=unit, squared_error=WINDOWS * (control_rmse * scale) ** 2, windows=30)
        )
        candidate.append(
            UnitError(
                unit=unit,
                squared_error=WINDOWS * (candidate_rmse * scale * draws.uniform(0.9, 1.1)) ** 2,
                windows=30,
            )
        )
    return PairedUnitErrors(control=tuple(control), candidate=tuple(candidate))


def test_a_true_reduction_is_found_with_its_whole_interval_above_zero() -> None:
    compared = PairedUnitBootstrap(resamples=2000, seed=1).compare(synthetic(30.0, 22.0, seed=3))

    assert compared.reduction == pytest.approx(8.0, abs=1.5)
    assert compared.relative_reduction == pytest.approx(8.0 / 30.0, abs=0.05)
    assert compared.interval.above_zero
    assert compared.interval.low < compared.reduction < compared.interval.high
    assert compared.p_value < 0.01
    assert compared.confirms(0.10)
    assert compared.distinguishable


def test_no_true_difference_leaves_zero_inside_the_interval() -> None:
    compared = PairedUnitBootstrap(resamples=2000, seed=1).compare(synthetic(30.0, 30.0, seed=5))

    assert abs(compared.reduction) < 1.0
    assert not compared.interval.excludes_zero
    assert compared.p_value > 0.2
    assert not compared.confirms(0.10)
    assert not compared.distinguishable


def test_the_same_seed_gives_the_same_interval_and_another_seed_a_near_one() -> None:
    compared = synthetic(30.0, 25.0, seed=7)
    first = PairedUnitBootstrap(resamples=1000, seed=1).compare(compared)
    again = PairedUnitBootstrap(resamples=1000, seed=1).compare(compared)
    other = PairedUnitBootstrap(resamples=1000, seed=2).compare(compared)

    assert first == again
    assert first.interval.low == pytest.approx(other.interval.low, abs=0.3)
    assert first.interval.high == pytest.approx(other.interval.high, abs=0.3)


def test_a_wider_level_gives_a_wider_interval() -> None:
    compared = synthetic(30.0, 25.0, seed=7)
    narrow = PairedUnitBootstrap(resamples=1000, seed=1, level=0.8).compare(compared).interval
    wide = PairedUnitBootstrap(resamples=1000, seed=1, level=0.99).compare(compared).interval

    assert wide.low < narrow.low
    assert narrow.high < wide.high
    assert (narrow.level, wide.level) == (0.8, 0.99)


def test_one_unit_resamples_to_itself_and_has_no_width() -> None:
    compared = PairedUnitErrors(
        control=(UnitError(unit=UnitKey("a"), squared_error=16.0, windows=1),),
        candidate=(UnitError(unit=UnitKey("a"), squared_error=4.0, windows=1),),
    )

    found = PairedUnitBootstrap(resamples=50, seed=1).compare(compared)

    assert (found.interval.low, found.interval.high) == (2.0, 2.0)
    # Every resample lies on one side, yet the p-value is not zero: the observed reduction
    # counts as one more, so fifty resamples can say no less than two in fifty-one.
    assert found.p_value == pytest.approx(2.0 / 51.0)


def test_a_bootstrap_without_a_resample_is_refused() -> None:
    with pytest.raises(InvalidPairedUnitBootstrapError, match="resamples must be positive"):
        PairedUnitBootstrap(resamples=0)


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_a_bootstrap_at_no_level_is_refused(level: float) -> None:
    with pytest.raises(InvalidPairedUnitBootstrapError, match="level must lie in"):
        PairedUnitBootstrap(level=level)
