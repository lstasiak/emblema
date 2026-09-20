from math import inf, nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidAdaptationScheduleError
from tests.evaluation.support import adaptation_schedule


def test_a_schedule_states_every_knob_of_the_learning() -> None:
    schedule = adaptation_schedule(
        epochs=30,
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_fraction=0.1,
        final_lr_fraction=0.01,
    )

    assert (schedule.epochs, schedule.batch_size) == (30, 16)
    assert (schedule.learning_rate, schedule.weight_decay) == (1e-4, 0.01)
    assert (schedule.warmup_fraction, schedule.final_lr_fraction) == (0.1, 0.01)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "epochs must be positive"),
        ("batch_size", 0, "batch_size must be positive"),
        ("min_steps", -1, "min_steps must not be negative"),
        ("learning_rate", 0.0, "learning_rate must be positive"),
        ("learning_rate", inf, "learning_rate must be positive and finite"),
        ("learning_rate", nan, "learning_rate must be positive and finite"),
        ("weight_decay", -0.01, "weight_decay must be finite and not negative"),
        ("weight_decay", inf, "weight_decay must be finite and not negative"),
        ("warmup_fraction", -0.1, "warmup_fraction must lie in"),
        ("warmup_fraction", 1.0, "warmup_fraction must lie in"),
        ("warmup_fraction", nan, "warmup_fraction must lie in"),
        ("final_lr_fraction", -0.1, "final_lr_fraction must lie in"),
        ("final_lr_fraction", 1.1, "final_lr_fraction must lie in"),
        ("final_lr_fraction", nan, "final_lr_fraction must lie in"),
    ],
)
def test_a_schedule_that_could_not_be_trained_is_refused(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(InvalidAdaptationScheduleError, match=message):
        adaptation_schedule(**{field: value})


def test_an_epoch_takes_one_step_per_batch_and_one_for_the_remainder() -> None:
    schedule = adaptation_schedule(batch_size=16)

    assert schedule.steps_per_epoch(16) == 1
    assert schedule.steps_per_epoch(17) == 2
    assert schedule.steps_per_epoch(200) == 13


def test_a_run_under_the_floor_of_steps_takes_as_many_whole_epochs_as_reach_it() -> None:
    schedule = adaptation_schedule(epochs=30, min_steps=2000, batch_size=16)

    assert schedule.epochs_over(50) == 500
    assert schedule.epochs_over(200) == 154
    assert schedule.epochs_over(1000) == 32
    assert schedule.epochs_over(2651) == 30


def test_a_run_over_the_floor_of_steps_keeps_its_epochs() -> None:
    assert adaptation_schedule(epochs=30, min_steps=0, batch_size=16).epochs_over(50) == 30
    assert adaptation_schedule(epochs=30, min_steps=100, batch_size=16).epochs_over(50) == 30


def test_the_rate_decays_over_the_epochs_the_floor_asks_for() -> None:
    schedule = adaptation_schedule(
        epochs=30, min_steps=2000, batch_size=16, warmup_fraction=0.1, final_lr_fraction=0.01
    )

    rate = schedule.learning_rate_schedule(200)

    assert (rate.warmup_steps, rate.total_steps) == (200, 2002)


def test_the_warmup_is_a_share_of_the_whole_run_in_steps() -> None:
    schedule = adaptation_schedule(
        epochs=30, batch_size=16, warmup_fraction=0.1, final_lr_fraction=0.01
    )

    rate = schedule.learning_rate_schedule(200)

    assert (rate.warmup_steps, rate.total_steps, rate.final_fraction) == (39, 390, 0.01)


def test_a_constant_rate_is_no_warmup_and_a_floor_at_the_peak() -> None:
    rate = adaptation_schedule(epochs=3, batch_size=2).learning_rate_schedule(4)

    assert (rate.warmup_steps, rate.final_fraction) == (0, 1.0)
    assert {rate.factor(step) for step in range(rate.total_steps)} == {1.0}


def test_a_run_of_one_step_keeps_that_step_to_decay_over() -> None:
    rate = adaptation_schedule(epochs=1, batch_size=8, warmup_fraction=0.9).learning_rate_schedule(
        3
    )

    assert (rate.warmup_steps, rate.total_steps) == (0, 1)


def test_a_sample_without_a_window_has_no_schedule() -> None:
    with pytest.raises(InvalidAdaptationScheduleError, match="hold a window"):
        adaptation_schedule().learning_rate_schedule(0)
