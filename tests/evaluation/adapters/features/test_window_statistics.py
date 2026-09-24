import math

import numpy as np
import pytest

from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.evaluation.adapters.features.support import timed, window

RISING = timed(2, [(2.0, 0.2), (4.0, 0.6), (6.0, 1.0)])
ONCE = timed(3, [(5.0, 0.5)])
STATIC = [Token(1, 7.0, 0.0, 0.0, timeless=True)]


def statistic(statistics: WindowStatistics, channel: int, name: str) -> float:
    row = int(np.flatnonzero(statistics.channels == channel)[0])
    return float(statistics.rows[row, WindowStatistics.NAMES.index(name)])


def test_the_channels_a_window_holds_are_reported_ascending() -> None:
    statistics = WindowStatistics.of(window(RISING, ONCE, STATIC))

    assert list(statistics.channels) == [1, 2, 3]


def test_a_channel_read_several_times_is_summarised_over_its_readings() -> None:
    statistics = WindowStatistics.of(window(RISING, ONCE))

    assert statistic(statistics, 2, "count") == 3.0
    assert statistic(statistics, 2, "mean") == pytest.approx(4.0)
    assert statistic(statistics, 2, "deviation") == pytest.approx(math.sqrt(8 / 3))
    assert statistic(statistics, 2, "minimum") == pytest.approx(2.0)
    assert statistic(statistics, 2, "maximum") == pytest.approx(6.0)


def test_the_ends_of_a_channel_are_its_first_and_last_reading_in_time() -> None:
    statistics = WindowStatistics.of(window(timed(2, [(9.0, 0.1), (1.0, 0.9)])))

    assert statistic(statistics, 2, "first") == pytest.approx(9.0)
    assert statistic(statistics, 2, "last") == pytest.approx(1.0)


def test_the_slope_is_the_rise_of_a_channel_over_the_window() -> None:
    statistics = WindowStatistics.of(window(RISING))

    assert statistic(statistics, 2, "slope") == pytest.approx(5.0)


def test_the_step_is_the_mean_move_between_consecutive_readings() -> None:
    statistics = WindowStatistics.of(window(timed(2, [(0.0, 0.2), (3.0, 0.4), (1.0, 0.9)])))

    assert statistic(statistics, 2, "step") == pytest.approx(2.5)


def test_a_channel_read_once_has_no_slope_and_no_step() -> None:
    statistics = WindowStatistics.of(window(RISING, ONCE))

    assert math.isnan(statistic(statistics, 3, "slope"))
    assert math.isnan(statistic(statistics, 3, "step"))


def test_a_static_feature_has_no_slope_because_it_carries_no_time() -> None:
    statistics = WindowStatistics.of(window(RISING, STATIC))

    assert statistic(statistics, 1, "mean") == pytest.approx(7.0)
    assert math.isnan(statistic(statistics, 1, "slope"))


def test_the_gap_is_the_mean_time_since_each_reading_s_predecessor() -> None:
    statistics = WindowStatistics.of(window(RISING))

    assert statistic(statistics, 2, "gap") == pytest.approx((0.2 + 0.4 + 0.4) / 3)


def test_the_moves_of_one_channel_are_not_counted_against_another() -> None:
    apart = window(timed(2, [(0.0, 0.1), (1.0, 0.2)]), timed(5, [(100.0, 0.3), (101.0, 0.4)]))
    statistics = WindowStatistics.of(apart)

    assert statistic(statistics, 2, "step") == pytest.approx(1.0)
    assert statistic(statistics, 5, "step") == pytest.approx(1.0)


def test_a_window_of_a_single_reading_is_summarised_without_complaint() -> None:
    statistics = WindowStatistics.of(TokenWindow.of(ONCE))

    assert statistic(statistics, 3, "count") == 1.0
    assert statistic(statistics, 3, "deviation") == pytest.approx(0.0)
