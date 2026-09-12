import pytest

from emblema.catalog.domain.exceptions import InvalidObservationError
from emblema.catalog.domain.measurements.observation import Observation
from tests.catalog.domain.support import BLANK_OR_PADDED, NON_FINITE


@pytest.mark.parametrize("channel", BLANK_OR_PADDED)
def test_observation_rejects_blank_or_padded_channel(channel: str) -> None:
    with pytest.raises(InvalidObservationError, match="non-blank"):
        Observation(channel, 0.0, 1.0)


@pytest.mark.parametrize("time", NON_FINITE)
def test_observation_time_is_finite(time: float) -> None:
    with pytest.raises(InvalidObservationError, match="time"):
        Observation("T2", time, 1.0)


@pytest.mark.parametrize("value", NON_FINITE)
def test_observation_value_is_finite_because_a_missing_value_is_no_observation(
    value: float,
) -> None:
    with pytest.raises(InvalidObservationError, match="value"):
        Observation("T2", 0.0, value)
