import pytest

from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.exceptions import InvalidChannelStatisticsError


def test_normalisation_is_the_z_score() -> None:
    assert ChannelStatistics(10, mean=4.0, std=2.0).normalise(7.0) == 1.5


def test_a_channel_that_never_varied_keeps_deviations_in_raw_units() -> None:
    assert ChannelStatistics(10, mean=518.67, std=0.0).normalise(518.68) == pytest.approx(0.01)


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
