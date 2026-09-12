import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.exceptions import InvalidChannelStatisticsError

values = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
# A spread is either zero, because the channel never varied, or a spread of the very values it was
# fitted on. Welford cannot return one hundreds of orders of magnitude below those values, and
# normalising against such a spread overflows to infinity — a combination the system never meets.
spreads = st.one_of(
    st.just(0.0), st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
)


def test_normalisation_is_the_z_score() -> None:
    assert ChannelStatistics(10, mean=4.0, std=2.0).normalise(7.0) == 1.5


def test_a_channel_that_never_varied_keeps_deviations_in_raw_units() -> None:
    assert ChannelStatistics(10, mean=518.67, std=0.0).normalise(518.68) == pytest.approx(0.01)


def test_denormalisation_puts_a_deviation_back_in_raw_units() -> None:
    assert ChannelStatistics(10, mean=4.0, std=2.0).denormalise(1.5) == 7.0


@given(value=values, mean=values, std=spreads)
def test_a_value_survives_the_round_trip_through_normalisation(
    value: float, mean: float, std: float
) -> None:
    statistics = ChannelStatistics(10, mean, std)
    assert statistics.denormalise(statistics.normalise(value)) == pytest.approx(value, abs=1e-6)


@pytest.mark.parametrize("count", [0, -1])
def test_statistics_come_from_at_least_one_value(count: int) -> None:
    with pytest.raises(InvalidChannelStatisticsError, match="count"):
        ChannelStatistics(count, 0.0, 1.0)


@pytest.mark.parametrize("mean", [float("nan"), float("inf")])
def test_mean_is_finite(mean: float) -> None:
    with pytest.raises(InvalidChannelStatisticsError, match="mean"):
        ChannelStatistics(1, mean, 1.0)


@pytest.mark.parametrize("std", [-0.1, float("nan"), float("inf")])
def test_std_is_finite_and_non_negative(std: float) -> None:
    with pytest.raises(InvalidChannelStatisticsError, match="std"):
        ChannelStatistics(1, 0.0, std)
