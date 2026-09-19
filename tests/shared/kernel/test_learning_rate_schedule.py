import math
from itertools import pairwise

import pytest

from emblema.shared.kernel.exceptions import InvalidLearningRateScheduleError
from emblema.shared.kernel.learning_rate_schedule import LearningRateSchedule


def schedule(*, warmup: int = 10, total: int = 110, final: float = 0.01) -> LearningRateSchedule:
    return LearningRateSchedule(warmup_steps=warmup, total_steps=total, final_fraction=final)


def test_the_warmup_climbs_to_the_peak_without_a_step_at_zero() -> None:
    warming = schedule(warmup=4)

    assert [warming.factor(step) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]


def test_the_decay_starts_at_the_peak_and_ends_at_the_floor_on_the_last_step() -> None:
    decaying = schedule(warmup=10, total=110, final=0.01)

    assert decaying.factor(10) == 1.0
    assert decaying.factor(109) == pytest.approx(0.01)
    assert decaying.factor(500) == pytest.approx(0.01)


def test_the_decay_passes_halfway_between_peak_and_floor_at_its_midpoint() -> None:
    decaying = schedule(warmup=0, total=101, final=0.2)

    assert decaying.factor(50) == pytest.approx(0.2 + 0.8 * 0.5)


def test_the_factor_never_rises_once_the_warmup_is_over() -> None:
    decaying = schedule(warmup=7, total=53, final=0.05)
    factors = [decaying.factor(step) for step in range(7, 60)]

    assert all(later <= earlier for earlier, later in pairwise(factors))


def test_a_floor_of_one_without_warmup_is_the_constant_rate() -> None:
    constant = schedule(warmup=0, total=30, final=1.0)

    assert {constant.factor(step) for step in range(40)} == {1.0}


def test_a_run_of_one_decaying_step_takes_it_at_the_peak() -> None:
    short = schedule(warmup=2, total=3, final=0.0)

    assert [short.factor(step) for step in range(4)] == [0.5, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("warmup", "total", "final", "field"),
    [
        (0, 0, 0.1, "total_steps"),
        (-1, 10, 0.1, "warmup_steps"),
        (10, 10, 0.1, "warmup_steps"),
        (0, 10, -0.1, "final_fraction"),
        (0, 10, 1.1, "final_fraction"),
        (0, 10, math.nan, "final_fraction"),
    ],
)
def test_a_schedule_breaking_its_invariants_is_rejected(
    warmup: int, total: int, final: float, field: str
) -> None:
    with pytest.raises(InvalidLearningRateScheduleError, match=field):
        schedule(warmup=warmup, total=total, final=final)


def test_a_negative_step_is_rejected() -> None:
    with pytest.raises(InvalidLearningRateScheduleError, match="step"):
        schedule().factor(-1)


def test_the_error_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        schedule(total=0)
