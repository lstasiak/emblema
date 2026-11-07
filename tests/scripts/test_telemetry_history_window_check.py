"""The history window check: hours concatenated as a window cut whole holds them, on CPU."""

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.joined_cells import JoinedCells
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.telemetry_history_window_check import HOUR_BINS, concatenated, hour_cells


def hour(*readings: tuple[int, float, float]) -> TokenWindow:
    return TokenWindow.of([Token(channel_id=c, value=v, time=t, gap=0.0) for c, v, t in readings])


def test_the_hours_take_their_share_of_the_window_and_count_their_gaps_anew() -> None:
    window = concatenated([hour((1, 2.0, 0.5)), None, hour((1, 3.0, 0.25), (2, 1.0, 0.75))])

    assert window.channel_ids == (1, 1, 2)
    assert window.times == pytest.approx((0.5 / 3, 2.25 / 3, 2.75 / 3))
    assert window.gaps == pytest.approx((0.5 / 3, 1.75 / 3, 2.75 / 3))


def test_the_hours_laid_once_and_joined_are_the_window_laid_whole() -> None:
    hours = [hour((1, 2.0, 0.5)), None, hour((1, 3.0, 0.25), (2, 1.0, 0.95))]
    channels = (1, 2)

    joined = JoinedCells(3).join(hour_cells(hours, channels))
    whole = BinnedTokens(3, channels=channels).lay([concatenated(hours)])[0].features

    assert hour_cells(hours, channels).shape == (2, 3 * HOUR_BINS, BinnedTokens.N_FEATURES)
    np.testing.assert_allclose(joined, whole.reshape(3, 2, -1).transpose(1, 0, 2), atol=1e-12)
