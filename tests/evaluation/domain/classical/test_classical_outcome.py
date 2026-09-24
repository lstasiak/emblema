from dataclasses import replace
from typing import Any

import pytest

from emblema.evaluation.domain.classical.classical_outcome import ClassicalOutcome
from emblema.evaluation.domain.exceptions import InvalidClassicalOutcomeError
from emblema.evaluation.domain.identifiers import UnitKey
from tests.evaluation.support import prediction

ANSWERS = (
    prediction("a", 0, 10.0, 8.0),
    prediction("a", 1, 10.0, 14.0),
    prediction("b", 0, 4.0, 4.0),
)


def outcome(**overrides: Any) -> ClassicalOutcome:
    """An outcome that holds together; anything named is replaced afterwards."""
    stated = ClassicalOutcome(predictions=ANSWERS, seconds=1.5, artifact=None)
    return replace(stated, **overrides)


def test_a_fit_that_answered_nothing_is_refused() -> None:
    with pytest.raises(InvalidClassicalOutcomeError, match="at least one window"):
        outcome(predictions=())


def test_a_window_answered_twice_is_refused() -> None:
    with pytest.raises(InvalidClassicalOutcomeError, match="predicted twice"):
        outcome(predictions=(ANSWERS[0], ANSWERS[0]))


@pytest.mark.parametrize("seconds", [-0.1, float("inf"), float("nan")])
def test_a_time_that_is_negative_or_not_a_number_is_refused(seconds: float) -> None:
    with pytest.raises(InvalidClassicalOutcomeError, match="finite and not negative"):
        outcome(seconds=seconds)


def test_the_error_is_accumulated_per_unit_in_unit_order() -> None:
    errors = outcome().by_unit()

    assert [error.unit for error in errors] == [UnitKey("a"), UnitKey("b")]
    assert errors[0].squared_error == pytest.approx(20.0)
    assert errors[0].windows == 2
    assert errors[1].squared_error == pytest.approx(0.0)


def test_the_error_over_every_window_is_the_root_of_their_mean_square() -> None:
    assert outcome().rmse == pytest.approx((20.0 / 3) ** 0.5)
