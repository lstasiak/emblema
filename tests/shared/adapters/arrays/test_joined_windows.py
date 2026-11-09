"""Windows joined with those before them: what is gathered where, and what is refused."""

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.joined_windows import JoinedWindows
from emblema.shared.kernel.tokens import Token, TokenWindow


def hour(value: float, channel: int = 1) -> TokenWindow:
    return TokenWindow.of([Token(channel_id=channel, value=value, time=0.5, gap=0.5)])


def joined(ends: list[float], axes: list[str] | None = None, history: int = 2) -> JoinedWindows:
    hours = [hour(float(at), channel=1 + at % 2) for at in range(len(ends))]
    return JoinedWindows.of(
        (hours[:1], hours[1:]),
        axes=axes or ["a"] * len(ends),
        starts=[end - 1.0 for end in ends],
        ends=ends,
        history=history,
        bins=history,
    )


def test_a_window_reads_the_one_before_it_on_its_axis_and_nothing_across_axes() -> None:
    windows = joined([1.0, 2.0, 2.0], axes=["a", "a", "b"])

    second, other = windows[1], windows[2]

    assert second.rows.tolist() == [1, 2]
    assert second.cells.shape == (2, 2, BinnedTokens.N_FEATURES)
    assert second.cells[0, 0, BinnedTokens.PRESENCE] == 1.0
    assert second.cells[1, 1, BinnedTokens.PRESENCE] == 1.0
    assert not other.cells[:, 0, BinnedTokens.PRESENCE].any()


def test_windows_asked_by_position_share_the_cells_laid_once() -> None:
    windows = joined([1.0, 2.0, 3.0])

    picked = windows.at([2, 0])

    assert len(picked) == 2
    assert np.array_equal(picked[0].cells, windows[2].cells)
    assert np.array_equal(windows[1:][0].cells, windows[1].cells)


@pytest.mark.parametrize(
    ("ends", "history", "match"),
    [([1.0, 1.5], 2, "overlap"), ([1.0, 2.0], 3, "do not cut"), ([1.0, 2.0], 0, "do not cut")],
)
def test_windows_that_cannot_be_joined_are_refused(
    ends: list[float], history: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        JoinedWindows.of(
            ([hour(1.0), hour(2.0)],),
            axes=["a", "a"],
            starts=[end - 1.0 for end in ends],
            ends=ends,
            history=history,
            bins=4,
        )


def test_no_windows_to_join_are_refused() -> None:
    with pytest.raises(ValueError, match="no windows"):
        JoinedWindows.of(((),), axes=[], starts=[], ends=[], history=1, bins=1)


def test_places_that_are_not_one_per_window_or_of_one_length_are_refused() -> None:
    with pytest.raises(ValueError, match="places"):
        JoinedWindows.of(([hour(1.0)],), axes=[], starts=[], ends=[], history=1, bins=1)
    with pytest.raises(ValueError, match="lengths"):
        JoinedWindows.of(
            ([hour(1.0), hour(2.0)],),
            axes=["a", "a"],
            starts=[0.0, 1.0],
            ends=[1.0, 3.0],
            history=1,
            bins=1,
        )


def test_a_window_holding_a_static_feature_is_refused() -> None:
    static = TokenWindow.of([Token(2, 1.0, 0.0, 0.0, timeless=True), Token(1, 1.0, 0.5, 0.5)])

    with pytest.raises(ValueError, match="static"):
        JoinedWindows.of(([static],), axes=["a"], starts=[0.0], ends=[1.0], history=1, bins=1)
