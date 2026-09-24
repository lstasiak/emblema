import math

import numpy as np
import pytest

from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.window_spectrum import WindowSpectrum
from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from tests.evaluation.adapters.features.support import timed, window

FEATURES = ChannelAggregatedFeatures()
RISING = timed(2, [(2.0, 0.2), (4.0, 0.6), (6.0, 1.0)])
FALLING = timed(5, [(9.0, 0.2), (7.0, 0.6), (5.0, 1.0)])
ONCE = timed(3, [(5.0, 0.5)])


def column(name: str, across: str) -> int:
    return WindowStatistics.NAMES.index(name) * len(FEATURES.ACROSS) + FEATURES.ACROSS.index(across)


def spectral(name: str, across: str) -> int:
    before = len(WindowStatistics.NAMES) * len(FEATURES.ACROSS)
    return (
        before
        + WindowSpectrum.NAMES.index(name) * len(FEATURES.ACROSS)
        + (FEATURES.ACROSS.index(across))
    )


def of_window(name: str) -> int:
    return FEATURES.width - len(FEATURES.OF_WINDOW) + FEATURES.OF_WINDOW.index(name)


def test_every_column_is_named_after_the_statistic_and_how_it_was_summarised() -> None:
    assert len(FEATURES.names()) == FEATURES.width
    assert FEATURES.names()[column("slope", "minimum")] == "minimum_slope"
    assert FEATURES.names()[spectral("centroid", "maximum")] == "maximum_spectrum_centroid"
    assert FEATURES.names()[of_window("channels")] == "window_channels"


def test_the_spectrum_is_summarised_over_the_channels_that_have_one() -> None:
    fast = timed(1, [(float(np.sin(2 * np.pi * 10 * k / 64)), k / 64) for k in range(1, 65)])
    slow = timed(2, [(float(np.sin(2 * np.pi * 2 * k / 64)), k / 64) for k in range(1, 65)])

    rows = FEATURES.of([window(fast, slow, ONCE)])

    assert rows[0, spectral("centroid", "minimum")] == pytest.approx(2.0, abs=0.5)
    assert rows[0, spectral("centroid", "maximum")] == pytest.approx(10.0, abs=0.5)


def test_the_width_is_the_same_whatever_the_corpus_it_read() -> None:
    narrow = FEATURES.of([window(RISING)])
    wide = FEATURES.of([window(RISING, FALLING, ONCE)])

    assert narrow.shape[1] == wide.shape[1] == FEATURES.width


def test_a_statistic_is_summarised_over_the_channels_that_have_one() -> None:
    rows = FEATURES.of([window(RISING, FALLING)])

    assert rows[0, column("slope", "mean")] == pytest.approx(0.0)
    assert rows[0, column("slope", "minimum")] == pytest.approx(-5.0)
    assert rows[0, column("slope", "maximum")] == pytest.approx(5.0)
    assert rows[0, column("slope", "deviation")] == pytest.approx(5.0)


def test_a_channel_without_a_statistic_is_left_out_of_that_summary() -> None:
    both = FEATURES.of([window(RISING, ONCE)])

    assert both[0, column("slope", "mean")] == pytest.approx(5.0)
    assert both[0, column("count", "mean")] == pytest.approx(2.0)


def test_a_statistic_no_channel_supports_stays_absent() -> None:
    rows = FEATURES.of([window(ONCE, timed(4, [(1.0, 0.3)]))])

    assert math.isnan(rows[0, column("slope", "mean")])
    assert math.isnan(rows[0, column("step", "maximum")])


def test_the_window_itself_is_described_beside_the_summary() -> None:
    rows = FEATURES.of([window(RISING, FALLING, ONCE)])

    assert rows[0, of_window("tokens")] == 7.0
    assert rows[0, of_window("channels")] == 3.0
    assert rows[0, of_window("timeless")] == 0.0


def test_two_corpora_of_different_layouts_produce_rows_that_can_be_stacked() -> None:
    eight = window(
        *(timed(channel, [(float(channel), 0.2), (0.0, 0.9)]) for channel in range(1, 9))
    )

    stacked = np.vstack((FEATURES.of([eight]), FEATURES.of([window(RISING), window(FALLING)])))

    assert stacked.shape == (3, FEATURES.width)


def test_the_rows_come_back_in_the_order_the_windows_were_given() -> None:
    rows = FEATURES.of([window(RISING), window(FALLING)])

    assert rows[0, column("slope", "mean")] == pytest.approx(5.0)
    assert rows[1, column("slope", "mean")] == pytest.approx(-5.0)
