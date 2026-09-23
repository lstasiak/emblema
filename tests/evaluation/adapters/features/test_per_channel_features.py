import math

import numpy as np
import pytest

from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from tests.evaluation.adapters.features.support import timed, window

RISING = timed(2, [(2.0, 0.2), (4.0, 0.6), (6.0, 1.0)])
ONCE = timed(3, [(5.0, 0.5)])
STATISTICS = len(WindowStatistics.NAMES)


def column(features: PerChannelFeatures, channel: int, name: str) -> int:
    return (channel - 1) * STATISTICS + WindowStatistics.NAMES.index(name)


def test_a_window_becomes_one_block_of_statistics_per_channel_of_the_corpus() -> None:
    features = PerChannelFeatures(4)

    assert features.width == 4 * STATISTICS
    assert len(features.names()) == features.width


def test_every_column_is_named_after_the_channel_and_the_statistic_it_holds() -> None:
    assert PerChannelFeatures(2).names()[STATISTICS] == "channel_2_count"


def test_the_statistics_of_a_channel_land_in_that_channel_s_block() -> None:
    features = PerChannelFeatures(3)

    rows = features.of([window(RISING, ONCE)])

    assert rows[0, column(features, 2, "count")] == 3.0
    assert rows[0, column(features, 2, "slope")] == pytest.approx(5.0)
    assert rows[0, column(features, 3, "mean")] == pytest.approx(5.0)


def test_a_channel_the_window_does_not_hold_is_counted_as_nothing_seen() -> None:
    features = PerChannelFeatures(3)

    rows = features.of([window(RISING)])

    assert rows[0, column(features, 1, "count")] == 0.0
    assert math.isnan(rows[0, column(features, 1, "mean")])
    assert math.isnan(rows[0, column(features, 3, "maximum")])


def test_the_rows_come_back_in_the_order_the_windows_were_given() -> None:
    features = PerChannelFeatures(3)

    rows = features.of([window(ONCE), window(RISING)])

    assert rows.shape == (2, features.width)
    assert rows[0, column(features, 3, "count")] == 1.0
    assert rows[1, column(features, 3, "count")] == 0.0


def test_a_window_of_a_wider_corpus_than_the_one_declared_is_refused() -> None:
    with pytest.raises(UnreadableTaskCorpusError, match="holds channel 3 and the corpus names 2"):
        PerChannelFeatures(2).of([window(RISING, ONCE)])


def test_a_corpus_without_a_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        PerChannelFeatures(0)


def test_the_width_binds_the_reading_to_one_corpus() -> None:
    widths = {PerChannelFeatures(channels).width for channels in (8, 24, 44)}

    assert len(widths) == 3


def test_nothing_is_left_uninitialised_between_two_windows_of_different_channels() -> None:
    features = PerChannelFeatures(3)

    rows = features.of([window(RISING), window(ONCE)])

    assert np.array_equal(rows[:, column(features, 2, "count")], [3.0, 0.0])
