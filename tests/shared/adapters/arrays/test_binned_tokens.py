import math
from typing import Any

import pytest

from emblema.shared.kernel.exceptions import InvalidTokenWindowError
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID, Token, TokenWindow

np = pytest.importorskip("numpy")

from numpy.typing import NDArray  # noqa: E402

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.arrays.token_batch import TokenBatch  # noqa: E402

pytestmark = pytest.mark.ml

STATIC = Token(9, 1.5, 0.0, 0.0, timeless=True)
# Channel 1 read at 0.10, 0.20 and 0.90, channel 2 at 0.30 and at the window's very end.
READINGS = [
    Token(1, 3.0, 0.10, 0.10),
    Token(1, 5.0, 0.20, 0.10),
    Token(2, -1.0, 0.30, 0.30),
    Token(1, 7.0, 0.90, 0.70),
    Token(2, 2.0, 1.0, 0.70),
]
WINDOW = TokenWindow.of([STATIC, *READINGS])
PRESENCE, COUNT, MEAN, MINIMUM, MAXIMUM, FIRST, LAST, OFFSET, SINCE = range(9)


def batched(layout: BinnedTokens, *windows: TokenWindow, dtype: Any = np.float32) -> TokenBatch:
    return layout.batch(layout.lay(list(windows)), dtype=dtype)


def cells(window: TokenWindow, layout: BinnedTokens) -> NDArray[Any]:
    return batched(layout, window, dtype=np.float64).features[0]


def test_the_layout_names_its_features_in_the_order_the_axis_holds_them() -> None:
    assert BinnedTokens(4).features == BinnedTokens.N_FEATURES == 9
    assert BinnedTokens.FEATURE_NAMES[PRESENCE] == "presence"
    assert BinnedTokens.FEATURE_NAMES[SINCE] == "since"


def test_cells_stand_in_canonical_order_timed_at_the_bins_centres() -> None:
    batch = batched(BinnedTokens(4), WINDOW)

    assert batch.channel_ids.tolist() == [[9, 1, 2, 1, 2, 1, 2, 1, 2]]
    assert batch.timestamps[0].tolist() == pytest.approx(
        [0.0, 0.125, 0.125, 0.375, 0.375, 0.625, 0.625, 0.875, 0.875]
    )
    assert batch.timeless.tolist() == [[True] + [False] * 8]
    assert not batch.padding_mask.any()
    assert batch.features.shape == (1, 9, BinnedTokens.N_FEATURES)


def test_an_observed_cell_summarises_the_readings_that_fell_in_its_bin() -> None:
    laid = cells(WINDOW, BinnedTokens(4))

    first_bin_of_channel_1 = laid[1]
    assert first_bin_of_channel_1.tolist() == pytest.approx(
        [1.0, math.log1p(2), 4.0, 3.0, 5.0, 3.0, 5.0, 0.8, 0.0]
    )
    second_bin_of_channel_2 = laid[4]
    assert second_bin_of_channel_2.tolist() == pytest.approx(
        [1.0, math.log1p(1), -1.0, -1.0, -1.0, -1.0, -1.0, 0.2, 0.0]
    )


def test_an_empty_cell_carries_the_channels_last_value_and_counts_the_bins_since() -> None:
    laid = cells(WINDOW, BinnedTokens(4))

    second_bin_of_channel_1 = laid[3]
    assert second_bin_of_channel_1.tolist() == pytest.approx(
        [0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.25]
    )
    third_bin_of_channel_1 = laid[5]
    assert third_bin_of_channel_1[SINCE] == pytest.approx(0.5)
    assert third_bin_of_channel_1[LAST] == 5.0


def test_before_a_channels_first_reading_its_cells_hold_zero_and_count_from_the_start() -> None:
    laid = cells(WINDOW, BinnedTokens(4))

    first_bin_of_channel_2 = laid[2]
    assert first_bin_of_channel_2.tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
    )


def test_a_reading_at_the_windows_end_falls_in_the_last_bin_with_an_offset_of_one() -> None:
    laid = cells(WINDOW, BinnedTokens(4))

    last_bin_of_channel_2 = laid[8]
    assert last_bin_of_channel_2[PRESENCE] == 1.0
    assert last_bin_of_channel_2[OFFSET] == pytest.approx(1.0)


def test_a_static_feature_is_one_timeless_cell_holding_its_value() -> None:
    laid = cells(WINDOW, BinnedTokens(4))

    assert laid[0].tolist() == pytest.approx(
        [1.0, math.log1p(1), 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0]
    )


def test_one_bin_summarises_the_whole_window_per_channel() -> None:
    laid = cells(WINDOW, BinnedTokens(1))

    assert laid[1].tolist() == pytest.approx(
        [1.0, math.log1p(3), 5.0, 3.0, 7.0, 3.0, 7.0, 0.9, 0.0]
    )
    assert laid[2].tolist() == pytest.approx(
        [1.0, math.log1p(2), 0.5, -1.0, 2.0, -1.0, 2.0, 1.0, 0.0]
    )


def test_the_order_the_tokens_arrive_in_changes_nothing() -> None:
    shuffled = TokenWindow.of(list(reversed([STATIC, *READINGS])))

    assert np.array_equal(cells(shuffled, BinnedTokens(4)), cells(WINDOW, BinnedTokens(4)))


def test_over_the_corpus_every_channel_gets_a_row_whether_the_window_holds_it_or_not() -> None:
    layout = BinnedTokens(2, channels=[1, 2, 5], statics=[9, 11])

    batch = batched(layout, WINDOW, dtype=np.float64)

    assert layout.over_the_corpus
    assert batch.channel_ids.tolist() == [[9, 11, 1, 2, 5, 1, 2, 5]]
    assert batch.timeless.tolist() == [[True, True] + [False] * 6]
    absent_static, never_read = batch.features[0, 1], batch.features[0, 4]
    assert absent_static.tolist() == [0.0] * 9
    assert never_read.tolist() == pytest.approx([0.0] * 8 + [0.5])
    assert batch.features[0, 7, SINCE] == pytest.approx(1.0)


def test_over_the_corpus_a_window_holding_another_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="channel 2, which the corpus does not name"):
        BinnedTokens(2, channels=[1]).lay([WINDOW])
    with pytest.raises(ValueError, match="static feature 9, which the corpus does not name"):
        BinnedTokens(2, channels=[1, 2]).lay([WINDOW])


def test_over_the_windows_own_channels_nothing_is_added() -> None:
    layout = BinnedTokens(2)

    assert not layout.over_the_corpus
    assert batched(layout, WINDOW).channel_ids.tolist() == [[9, 1, 2, 1, 2]]


def test_windows_are_padded_to_the_one_with_the_most_cells() -> None:
    short = TokenWindow.of([Token(1, 0.5, 0.5, 0.5)])

    batch = batched(BinnedTokens(2), WINDOW, short)

    assert batch.padding_mask.tolist() == [[False] * 5, [False, False, True, True, True]]
    assert batch.channel_ids[1].tolist() == [1, 1] + [PADDING_CHANNEL_ID] * 3
    assert (batch.features[1, 2:] == 0).all()
    assert batch.features.dtype == np.float32


@pytest.mark.parametrize(
    ("bins", "channels", "statics", "message"),
    [
        (0, None, (), "a layout has bins"),
        (2, None, (9,), "named with the corpus's channels"),
        (2, (1, 0), (), "a channel identifier is positive"),
        (2, (1, 1), (), "named once"),
        (2, (1,), (1,), "named once"),
    ],
)
def test_a_layout_that_cannot_be_laid_is_refused(
    bins: int, channels: tuple[int, ...] | None, statics: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BinnedTokens(bins, channels=channels, statics=statics)


def test_a_batch_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        BinnedTokens(2).batch([])


def test_a_static_feature_named_twice_is_summarised_as_a_bin_would_be() -> None:
    twice = TokenWindow.of(
        [
            Token(9, 1.0, 0.0, 0.0, timeless=True),
            Token(9, 3.0, 0.0, 0.0, timeless=True),
            *READINGS,
        ]
    )

    laid = cells(twice, BinnedTokens(2))

    assert laid[0].tolist() == pytest.approx(
        [1.0, math.log1p(2), 2.0, 1.0, 3.0, 1.0, 3.0, 0.0, 0.0]
    )


def test_a_binned_batch_spells_no_windows() -> None:
    with pytest.raises(InvalidTokenWindowError, match="spells no windows"):
        batched(BinnedTokens(2), WINDOW).windows()


def test_the_value_features_of_a_cell_are_its_mean_extremes_and_ends() -> None:
    names = BinnedTokens.FEATURE_NAMES[BinnedTokens(2).value_features]

    assert names == ("mean", "minimum", "maximum", "first", "last")
