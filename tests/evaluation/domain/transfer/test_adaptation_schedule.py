from math import inf, nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidAdaptationScheduleError
from tests.evaluation.support import adaptation_schedule


def test_a_schedule_states_every_knob_of_the_learning() -> None:
    schedule = adaptation_schedule(epochs=30, batch_size=16, learning_rate=1e-4, weight_decay=0.01)

    assert (schedule.epochs, schedule.batch_size) == (30, 16)
    assert (schedule.learning_rate, schedule.weight_decay) == (1e-4, 0.01)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "epochs must be positive"),
        ("batch_size", 0, "batch_size must be positive"),
        ("learning_rate", 0.0, "learning_rate must be positive"),
        ("learning_rate", inf, "learning_rate must be positive and finite"),
        ("learning_rate", nan, "learning_rate must be positive and finite"),
        ("weight_decay", -0.01, "weight_decay must be finite and not negative"),
        ("weight_decay", inf, "weight_decay must be finite and not negative"),
    ],
)
def test_a_schedule_that_could_not_be_trained_is_refused(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(InvalidAdaptationScheduleError, match=message):
        adaptation_schedule(**{field: value})
