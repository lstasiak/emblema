from collections.abc import Callable
from statistics import NormalDist

import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.channels.channel_statistics import (
    QUANTILE_KNOTS,
    RANK_FLOOR,
    ROBUST_BOUND,
    ChannelStatistics,
)
from emblema.catalog.domain.channels.normalisation import Normalisation
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


EVEN = tuple(float(i) for i in range(QUANTILE_KNOTS))


def robust(median: float = 4.0, mad: float = 1.0, std: float = 3.0) -> ChannelStatistics:
    return ChannelStatistics(10, 5.0, std, Normalisation.ROBUST, median=median, mad=mad)


def by_quantile(quantiles: tuple[float, ...] = EVEN) -> ChannelStatistics:
    return ChannelStatistics(10, 500.0, 300.0, Normalisation.QUANTILE, quantiles=quantiles)


def test_the_robust_score_measures_from_the_median_in_robust_deviations() -> None:
    assert robust().normalise(4.0 + 1.4826) == pytest.approx(1.0)


def test_the_robust_score_is_bounded() -> None:
    assert robust().normalise(1e6) == ROBUST_BOUND
    assert robust().normalise(-1e6) == -ROBUST_BOUND


def test_a_channel_mostly_at_one_value_measures_robust_scores_in_standard_deviations() -> None:
    assert robust(mad=0.0, std=2.0).normalise(8.0) == pytest.approx(2.0)


def test_the_robust_score_comes_back_as_the_raw_value() -> None:
    statistics = robust()

    assert statistics.denormalise(statistics.normalise(6.3)) == pytest.approx(6.3)


def test_the_median_knot_has_a_quantile_score_of_zero() -> None:
    assert by_quantile().normalise(500.0) == pytest.approx(0.0, abs=1e-12)


def test_a_value_beyond_every_knot_takes_the_bounded_rank() -> None:
    assert by_quantile().normalise(1e9) == pytest.approx(NormalDist().inv_cdf(1 - RANK_FLOOR))
    assert by_quantile().normalise(-1e9) == pytest.approx(NormalDist().inv_cdf(RANK_FLOOR))


def test_a_value_equal_to_several_knots_takes_the_middle_of_their_ranks() -> None:
    # Half the knots at 0, half at 1: a channel recorded at two levels.
    two_levels = tuple(0.0 if i < 500 else 1.0 for i in range(QUANTILE_KNOTS))

    low, high = by_quantile(two_levels).normalise(0.0), by_quantile(two_levels).normalise(1.0)

    assert low == pytest.approx(-high, rel=1e-2)
    assert low < 0 < high


@given(first=values, second=values)
def test_the_quantile_score_never_reverses_an_order(first: float, second: float) -> None:
    statistics = by_quantile(tuple(float(i) - 500.0 for i in range(QUANTILE_KNOTS)))
    low, high = sorted((first, second))

    assert statistics.normalise(low) <= statistics.normalise(high)


def test_a_quantile_score_comes_back_between_its_knots() -> None:
    statistics = by_quantile()

    assert statistics.denormalise(statistics.normalise(123.4)) == pytest.approx(123.4, abs=1e-6)


STANDARD, ROBUST, QUANTILE = Normalisation.STANDARD, Normalisation.ROBUST, Normalisation.QUANTILE
MISMATCHED: dict[str, Callable[[], ChannelStatistics]] = {
    "standard with a median": lambda: ChannelStatistics(1, 0.0, 1.0, STANDARD, 1.0, 1.0),
    "robust without one": lambda: ChannelStatistics(1, 0.0, 1.0, ROBUST),
    "robust with quantiles": lambda: ChannelStatistics(1, 0.0, 1.0, ROBUST, 1.0, 1.0, EVEN),
    "quantile without them": lambda: ChannelStatistics(1, 0.0, 1.0, QUANTILE),
    "standard with quantiles": lambda: ChannelStatistics(1, 0.0, 1.0, STANDARD, quantiles=EVEN),
}


@pytest.mark.parametrize("build", MISMATCHED.values(), ids=MISMATCHED.keys())
def test_statistics_keep_exactly_the_facts_their_normalisation_reads(
    build: Callable[[], ChannelStatistics],
) -> None:
    with pytest.raises(InvalidChannelStatisticsError):
        build()


@pytest.mark.parametrize(
    "quantiles",
    [EVEN[:-1], (*EVEN[:-1], float("nan")), (*EVEN[:-2], 2000.0, 1999.0)],
    ids=["too few", "not finite", "decreasing"],
)
def test_quantiles_that_are_not_a_full_ordered_set_are_refused(
    quantiles: tuple[float, ...],
) -> None:
    with pytest.raises(InvalidChannelStatisticsError, match="quantile"):
        by_quantile(quantiles)


@pytest.mark.parametrize("mad", [-1.0, float("nan")])
def test_a_robust_deviation_that_is_negative_or_not_finite_is_refused(mad: float) -> None:
    with pytest.raises(InvalidChannelStatisticsError, match="mad"):
        robust(mad=mad)
