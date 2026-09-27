import pytest

from emblema.serving.domain.exceptions import (
    EmptyRequestError,
    InvalidInferenceLimitsError,
    TooManyWindowsError,
    WindowTooLongError,
)
from emblema.serving.domain.inference_limits import InferenceLimits
from tests.serving.support import limits


@pytest.mark.parametrize("field", ["max_windows_per_request", "max_tokens_per_window"])
@pytest.mark.parametrize("count", [0, -1])
def test_a_limit_is_positive(field: str, count: int) -> None:
    with pytest.raises(InvalidInferenceLimitsError, match=field):
        limits(**{field: count})


def test_a_request_within_the_limit_is_admitted() -> None:
    InferenceLimits(max_windows_per_request=2, max_tokens_per_window=3).admit_count(2)
    InferenceLimits(max_windows_per_request=2, max_tokens_per_window=3).admit_length(3)


def test_a_request_without_a_window_is_refused() -> None:
    with pytest.raises(EmptyRequestError):
        limits().admit_count(0)


def test_a_request_of_more_windows_than_admitted_is_refused_naming_both_counts() -> None:
    with pytest.raises(TooManyWindowsError, match="at most 4 windows, got 5"):
        limits().admit_count(5)


def test_a_window_longer_than_admitted_is_refused_naming_both_counts() -> None:
    with pytest.raises(WindowTooLongError, match="at most 16 tokens, got 17"):
        limits().admit_length(17)
