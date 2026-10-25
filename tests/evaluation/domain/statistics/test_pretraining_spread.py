from math import sqrt

import pytest

from emblema.evaluation.domain.exceptions import InvalidPretrainingSpreadError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.paired_reading import PairedReading
from emblema.evaluation.domain.statistics.pretraining_spread import PretrainingSpread


def pair(mean: float, *, standard_error: float, half: float) -> PairedReading:
    return PairedReading(
        mean=mean,
        standard_error=standard_error,
        interval=BootstrapInterval(low=mean - half, high=mean + half, level=0.95),
    )


def test_the_spread_is_the_squared_difference_less_its_noise_averaged_then_rooted() -> None:
    pairs = [
        pair(0.03, standard_error=0.004, half=0.0196),
        pair(-0.01, standard_error=0.002, half=0.0098),
    ]
    noise = (0.004**2 + 0.01**2) + (0.002**2 + 0.005**2)

    spread = PretrainingSpread.of(pairs, floor=0.014)

    assert spread.own == pytest.approx(sqrt((0.03**2 + 0.01**2 - noise) / 2))
    assert spread.value == spread.own > 0.014


def test_pairs_quieter_than_their_noise_leave_the_floor() -> None:
    spread = PretrainingSpread.of([pair(0.001, standard_error=0.005, half=0.01)], floor=0.014)

    assert spread.own == 0.0
    assert spread.value == 0.014


def test_a_spread_needs_a_pair_and_figures_in_range() -> None:
    with pytest.raises(InvalidPretrainingSpreadError, match="at least one pair"):
        PretrainingSpread.of([], floor=0.014)
    with pytest.raises(InvalidPretrainingSpreadError, match="floor"):
        PretrainingSpread(own=0.01, floor=-0.1)
