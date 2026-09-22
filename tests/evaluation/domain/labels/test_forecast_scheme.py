from math import inf, nan

import pytest

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme


def test_the_reading_is_taken_a_horizon_past_the_windows_end() -> None:
    assert ForecastScheme("s01", 12.0).instant_of(40.0) == 52.0
    assert ForecastScheme("s01", 0.0).instant_of(40.0) == 40.0


def test_the_target_is_the_exact_reading_itself() -> None:
    assert ForecastScheme("s01", 12.0).target(exact=-0.75) == -0.75


@pytest.mark.parametrize("channel", ["", " s01", "s01 "])
def test_a_blank_or_padded_channel_is_refused(channel: str) -> None:
    with pytest.raises(InvalidLabelSchemeError, match="channel"):
        ForecastScheme(channel, 12.0)


@pytest.mark.parametrize("horizon", [-1.0, inf, nan])
def test_a_horizon_that_is_negative_or_not_finite_is_refused(horizon: float) -> None:
    with pytest.raises(InvalidLabelSchemeError, match="horizon"):
        ForecastScheme("s01", horizon)
