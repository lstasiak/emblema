"""Cells joined from consecutive windows against the layout of their readings concatenated."""

import random

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.joined_cells import JoinedCells
from emblema.shared.kernel.tokens import Token, TokenWindow

CHANNELS = (1, 2, 3, 4)
# Six bins a window, as the satellites' hours are published.
HOUR_BINS = 6


def hour_window(draw: random.Random, *, channels: tuple[int, ...], readings: int) -> TokenWindow:
    """One published window of an hour: readings of the given channels at random instants."""
    tokens = [
        Token(
            channel_id=draw.choice(channels),
            value=round(draw.uniform(-3.0, 3.0), 3),
            time=draw.random(),
            gap=0.0,
        )
        for _ in range(readings)
    ]
    return TokenWindow.of(tokens)


def concatenated(hours: list[TokenWindow | None]) -> TokenWindow:
    """The hours as one window, each hour's times placed in its share of the whole."""
    count = len(hours)
    tokens = [
        Token(channel_id=t.channel_id, value=t.value, time=(at + t.time) / count, gap=0.0)
        for at, hour in enumerate(hours)
        if hour is not None
        for t in hour
    ]
    return TokenWindow.of(tokens)


def published_cells(hours: list[TokenWindow | None]) -> np.ndarray:
    """``[rows, hours * HOUR_BINS, features]``: each hour laid over the corpus's channels."""
    layout = BinnedTokens(HOUR_BINS, channels=CHANNELS)
    laid = []
    for hour in hours:
        if hour is None:
            laid.append(np.zeros((len(CHANNELS), HOUR_BINS, BinnedTokens.N_FEATURES)))
        else:
            cells = layout.lay([hour])[0].features
            laid.append(cells.reshape(HOUR_BINS, len(CHANNELS), -1).transpose(1, 0, 2))
    return np.concatenate(laid, axis=1)


@pytest.mark.parametrize(("hours", "bins"), [(1, 6), (6, 6), (6, 36), (24, 24), (4, 8), (3, 2)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_joined_cells_are_what_the_layout_makes_of_the_readings_concatenated(
    hours: int, bins: int, seed: int
) -> None:
    draw = random.Random(seed)
    windows: list[TokenWindow | None] = [
        None
        if draw.random() < 0.2
        else hour_window(
            draw, channels=draw.choice([(1, 2), (2, 3, 4), CHANNELS]), readings=draw.randint(1, 30)
        )
        for _ in range(hours)
    ]
    if all(w is None for w in windows):
        windows[-1] = hour_window(draw, channels=CHANNELS, readings=5)
    whole = concatenated(windows)

    derived = JoinedCells(bins).join(published_cells(windows))
    over_corpus = BinnedTokens(bins, channels=CHANNELS).lay([whole])[0]
    own = BinnedTokens(bins).lay([whole])[0]

    assert derived.shape == (len(CHANNELS), bins, BinnedTokens.N_FEATURES)
    expected = over_corpus.features.reshape(bins, len(CHANNELS), -1).transpose(1, 0, 2)
    np.testing.assert_allclose(derived, expected, atol=1e-9)
    rows = np.asarray(CHANNELS, dtype=np.int64)
    laid_corpus = JoinedCells.laid(derived, rows, own_channels=False)
    np.testing.assert_allclose(laid_corpus.features, over_corpus.features, atol=1e-9)
    assert laid_corpus.channel_ids.tolist() == over_corpus.channel_ids.tolist()
    np.testing.assert_allclose(laid_corpus.times, over_corpus.times)
    laid_own = JoinedCells.laid(derived, rows, own_channels=True)
    np.testing.assert_allclose(laid_own.features, own.features, atol=1e-9)
    assert laid_own.channel_ids.tolist() == own.channel_ids.tolist()
    assert not laid_own.timeless.any()


def test_cells_that_do_not_cut_into_the_bins_are_refused() -> None:
    with pytest.raises(ValueError, match="do not cut"):
        JoinedCells(5).join(np.zeros((2, 12, BinnedTokens.N_FEATURES)))
    with pytest.raises(ValueError, match="at least one bin"):
        JoinedCells(0)


def test_a_window_that_is_not_there_carries_the_last_reading_and_counts_since_it() -> None:
    cells = np.zeros((1, 3, BinnedTokens.N_FEATURES))
    cells[0, 0, BinnedTokens.PRESENCE] = 1.0
    cells[0, 0, BinnedTokens.LOG_COUNT] = np.log1p(2)
    cells[0, 0, 2:7] = 4.0

    joined = JoinedCells(3).join(cells)

    assert joined[0, :, BinnedTokens.PRESENCE].tolist() == [1.0, 0.0, 0.0]
    assert joined[0, 2, 2:7].tolist() == [4.0] * 5
    assert joined[0, :, 8].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3])
