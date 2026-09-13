import numpy as np
import pytest

pytest.importorskip("torch")

from emblema.pretraining.adapters.diagnostics.channel_series import ChannelSeries

pytestmark = pytest.mark.ml


def test_of_window_collects_each_channels_visible_tokens_in_time_order() -> None:
    channel_ids = np.array([2, 1, 2, 1, 2])
    times = np.array([0.9, 0.5, 0.1, 0.0, 0.5])
    values = np.array([9.0, 5.0, 1.0, 0.0, 5.5])
    visible = np.array([True, True, True, False, True])

    series = ChannelSeries.of_window(channel_ids, times, values, visible)

    assert set(series) == {1, 2}
    assert series[1].times.tolist() == [0.5]
    assert series[2].times.tolist() == [0.1, 0.5, 0.9]
    assert series[2].values.tolist() == [1.0, 5.5, 9.0]
    assert len(series[2]) == 3


def test_interpolation_is_exact_on_a_straight_line() -> None:
    line = ChannelSeries(np.array([0.0, 0.5, 1.0]), np.array([1.0, 2.0, 3.0]))

    assert line.interpolate(np.array([0.25, 0.75])).tolist() == pytest.approx([1.5, 2.5])


def test_beyond_the_visible_ends_the_end_value_is_carried_over() -> None:
    line = ChannelSeries(np.array([0.4, 0.6]), np.array([1.0, 3.0]))

    assert line.interpolate(np.array([0.0, 1.0])).tolist() == [1.0, 3.0]


def test_nearest_picks_the_closer_instant_on_either_side() -> None:
    line = ChannelSeries(np.array([0.0, 0.5, 1.0]), np.array([10.0, 20.0, 30.0]))

    assert line.nearest(np.array([0.2, 0.3, 0.5, 0.9, 1.0])).tolist() == [
        10.0,
        20.0,
        20.0,
        30.0,
        30.0,
    ]


def test_nearest_beyond_the_ends_is_the_end() -> None:
    line = ChannelSeries(np.array([0.4, 0.6]), np.array([1.0, 3.0]))

    assert line.nearest(np.array([0.0, 1.0])).tolist() == [1.0, 3.0]
