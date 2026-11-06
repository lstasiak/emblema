import numpy as np
import pytest

from emblema.evaluation.adapters.detection.channel_departures import ChannelDepartures
from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import OverlappingWindowsError
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens

F = BinnedTokens.N_FEATURES
PRESENCE, LOG_COUNT = BinnedTokens.PRESENCE, BinnedTokens.LOG_COUNT
MEAN_AT, MIN_AT, MAX_AT, LAST_AT = (
    BinnedTokens.FEATURE_NAMES.index(name) for name in ("mean", "minimum", "maximum", "last")
)
BINS = 6


def cell(bins: list[tuple[float, float, float, float, int] | None]) -> np.ndarray:
    """One channel's cells over the bins: (mean, min, max, last, readings) or an empty bin."""
    out = np.zeros((len(bins), F))
    for place, each in enumerate(bins):
        if each is None:
            continue
        mean, least, most, last, readings = each
        out[place, PRESENCE] = 1.0
        out[place, LOG_COUNT] = np.log1p(readings)
        out[place, MEAN_AT], out[place, MIN_AT], out[place, MAX_AT], out[place, LAST_AT] = (
            mean,
            least,
            most,
            last,
        )
    return out


def hourly_cells(values: list[float | None], readings: int = 2) -> np.ndarray:
    """One channel, one window per hour, every bin at the hour's value, or an empty hour."""
    out = np.zeros((len(values), 1, BINS, F))
    for hour, value in enumerate(values):
        if value is not None:
            out[hour, 0] = cell([(value, value - 0.5, value + 0.5, value, readings)] * BINS)
    return out


def test_a_window_is_summarised_over_its_present_spans_and_missing_where_none_is() -> None:
    cells = np.stack(
        [
            cell([(1.0, 0.0, 2.0, 1.5, 3), None, (3.0, 2.0, 4.0, 3.5, 1)]),
            cell([None, None, None]),
        ]
    )[None]

    summaries = ChannelDepartures.summaries_of(cells)

    assert summaries.shape == (1, 2, 5)
    np.testing.assert_allclose(summaries[0, 0], [2.0, 0.0, 4.0, 3.5, np.log1p(4.0)])
    assert np.isnan(summaries[0, 1]).all()


def test_the_spans_values_are_missing_where_the_span_holds_nothing() -> None:
    values = ChannelDepartures.bin_values_of(cell([(1.0, 0.0, 2.0, 1.5, 3), None])[None, None])

    assert values.shape == (1, 1, 2, 5)
    np.testing.assert_allclose(values[0, 0, 0], [1.0, 0.0, 2.0, 1.5, np.log1p(3.0)])
    assert np.isnan(values[0, 0, 1]).all()


def laid(summaries: np.ndarray) -> ChannelDepartures:
    return ChannelDepartures(
        history=ChannelHistory(),
        summaries=summaries,
        bins=None,
        before=np.full((len(summaries), 5), -1, dtype=np.int64),
    )


def test_the_floor_is_a_share_of_each_channels_spread_or_of_the_typical_one() -> None:
    summaries = np.full((5, 2, 5), np.nan)
    summaries[:, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]
    summaries[:, 1, 0] = 7.0
    summaries[:, 0, 1] = [0.0, 10.0, 20.0, 30.0, 40.0]

    floor = laid(summaries).floor_over(np.arange(5))

    mad_a, mad_b = 1.4826 * 1.0, 1.4826 * 10.0
    typical = np.median([mad_a, mad_b])
    assert floor[0, 0] == pytest.approx(0.1 * mad_a)
    assert floor[0, 1] == pytest.approx(0.1 * mad_b)
    assert floor[1, 0] == pytest.approx(0.1 * typical)
    assert floor[1, 1] == pytest.approx(0.1 * typical)
    assert laid(np.full((3, 1, 5), 2.0)).floor_over(np.arange(3)).tolist() == [[1.0] * 5]


def test_a_departure_is_the_distance_from_the_recent_median_over_the_floored_spread() -> None:
    # Window 4 reads windows 0-3: channel 0 at 9, 10, 11, 10 then 13; channel 1 constant at 5,
    # then one above, read over the floor; channel 2 absent from the window; channel 3 held in
    # one earlier window of four, fewer than the half it needs.
    summaries = np.full((5, 4, 5), np.nan)
    summaries[:4, 0, 0] = [9.0, 10.0, 11.0, 10.0]
    summaries[4, 0, 0] = 13.0
    summaries[:4, 1, 0] = 5.0
    summaries[4, 1, 0] = 6.0
    summaries[:4, 2, 0] = 1.0
    summaries[0, 3, 0] = 1.0
    summaries[4, 3, 0] = 100.0
    departures = ChannelDepartures(
        history=ChannelHistory(span=5),
        summaries=summaries,
        bins=None,
        before=np.array([[-1] * 4] * 4 + [[3, 2, 1, 0]], dtype=np.int64),
    )

    found = departures.of(np.array([4]), np.full((4, 5), 0.5))

    assert found.shape == (1, 4)
    assert found[0, 0] == pytest.approx(3.0 / max(1.4826 * 0.5, 0.5))
    assert found[0, 1] == pytest.approx(1.0 / 0.5)
    assert found[0, 2] == 0.0
    assert found[0, 3] == 0.0
    assert departures.present(np.array([4])).tolist() == [[True, True, False, True]]


@pytest.mark.parametrize("departure", list(DepartureCells))
def test_every_window_departs_from_the_windows_before_it_over_the_fitting_floor(
    departure: DepartureCells,
) -> None:
    values: list[float | None] = [1.0, 1.0, 1.0, 1.0, 1.0, None, 1.0, 4.0]
    cells = hourly_cells(values)
    history = ChannelHistory(span=4, departure=departure)

    departures = ChannelDepartures.laid(
        history,
        [cells[:3], cells[3:6], cells[6:]],
        ends_at=[float(h) for h in range(8)],
        units=["m1"] * 5 + ["m2"] * 3,
        window_length=1.0,
    )
    floor = departures.floor_over(np.arange(5))
    found = departures.of(np.arange(8), floor)

    # Nothing varies over the fitting windows, so the floor is the share of one; the last
    # window reads windows 4 and 6, window 5 being empty, and every value of it lies three
    # above theirs. Window 1 has one earlier window of the two it needs; window 5 is empty.
    assert floor.tolist() == [[1.0] * 5]
    assert departures.before[7].tolist() == [6, 5, 4]
    assert departures.before[1].tolist() == [0, -1, -1]
    assert found[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    assert departures.present(np.arange(8))[:, 0].tolist() == [v is not None for v in values]


def test_the_history_counts_earlier_windows_held_not_their_spans() -> None:
    # Two of three earlier windows hold a reading, in one span each: enough under the bins.
    cells = np.zeros((4, 1, BINS, F))
    cells[0, 0, 0] = cell([(1.0, 1.0, 1.0, 1.0, 1)])[0]
    cells[1, 0, 2] = cell([(1.0, 1.0, 1.0, 1.0, 1)])[0]
    cells[3, 0] = cell([(5.0, 5.0, 5.0, 5.0, 1)] * BINS)
    history = ChannelHistory(span=4, departure=DepartureCells.BINS)

    departures = ChannelDepartures.laid(
        history, [cells], ends_at=[0.0, 1.0, 2.0, 3.0], units=["m"] * 4, window_length=1.0
    )

    assert departures.of(np.array([3]), np.full((1, 5), 1.0))[0, 0] == pytest.approx(4.0)


def test_a_window_off_the_others_by_seconds_takes_its_place_in_their_history() -> None:
    # A month published 16 seconds off the hour, then one on it, on one axis.
    off = 16.0 / 3600.0
    ends = np.array([0.0 + off, 1.0 + off, 2.0 + off, 3.0, 4.0])

    before = ChannelDepartures.windows_before(ends, length=1.0, count=3)

    assert before[3].tolist() == [2, 1, 0]
    assert before[4].tolist() == [3, 2, 1]
    assert before[0].tolist() == [-1, -1, -1]


def test_a_missing_window_leaves_its_place_in_the_history_empty_in_any_order() -> None:
    ends = np.array([5.0, 0.0, 1.0, 3.0])

    before = ChannelDepartures.windows_before(ends, length=1.0, count=4)

    assert before[0].tolist() == [-1, 3, -1, 2]


def test_a_window_nearest_each_earlier_length_is_taken_where_windows_overlap() -> None:
    # Windows of ten every five: the history steps a whole length back each time.
    ends = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

    assert ChannelDepartures.windows_before(ends, length=10.0, count=2)[4].tolist() == [2, 0]


@pytest.mark.parametrize(
    ("ends", "units"),
    [
        ([0.0, 1.0, 0.5, 2.0], ["a", "a", "b", "b"]),
        ([3.0, 3.0], ["a", "b"]),
        ([0.0, 5.0, 2.0], ["a", "a", "b"]),
        ([0.0, 10.0, 1.0, 2.0, 3.0], ["a", "a", "b", "b", "c"]),
    ],
)
def test_units_whose_windows_interleave_in_time_are_refused(
    ends: list[float], units: list[str]
) -> None:
    with pytest.raises(OverlappingWindowsError, match="interleave in time"):
        ChannelDepartures.refuse_interleaved(np.array(ends), units)


def test_units_one_after_another_on_one_axis_are_read_as_one_history() -> None:
    ChannelDepartures.refuse_interleaved(np.array([0.0, 1.0, 2.0, 3.0]), ["a", "a", "b", "b"])


def test_cells_and_ends_naming_different_windows_are_refused() -> None:
    with pytest.raises(ValueError, match="2 windows were laid for 3 ends"):
        ChannelDepartures.laid(
            ChannelHistory(),
            [hourly_cells([1.0, 1.0])],
            ends_at=[0.0, 1.0, 2.0],
            units=["m"] * 3,
            window_length=1.0,
        )


def test_no_window_has_no_history_and_no_departure() -> None:
    assert ChannelDepartures.windows_before(np.array([]), length=1.0, count=3).shape == (0, 3)
    departures = laid(np.full((2, 3, 5), 1.0))
    assert departures.of(np.array([], dtype=np.int64), np.ones((3, 5))).shape == (0, 3)


def test_interleaved_units_are_refused_before_a_cell_is_laid() -> None:
    cells = iter([hourly_cells([1.0, 1.0])])

    with pytest.raises(OverlappingWindowsError, match="interleave in time"):
        ChannelDepartures.laid(
            ChannelHistory(), cells, ends_at=[0.0, 0.0], units=["a", "b"], window_length=1.0
        )
    with pytest.raises(ValueError, match="2 ends were given for 1 units"):
        ChannelDepartures.laid(
            ChannelHistory(), cells, ends_at=[0.0, 1.0], units=["a"], window_length=1.0
        )
    # Neither refusal laid the cells it was handed.
    assert next(cells).shape == (2, 1, BINS, F)
