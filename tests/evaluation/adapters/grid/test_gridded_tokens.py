import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.evaluation.adapters.grid.gridded_tokens import GriddedTokens
from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.evaluation.adapters.features.support import timed, window


def test_a_channel_carries_its_latest_reading_from_its_first_step_on() -> None:
    laid = GriddedTokens(5).of([window(timed(1, [(3.0, 0.25), (7.0, 0.65)]))])[0]

    assert list(laid) == [
        Token(1, 3.0, 0.2, 0.0),
        Token(1, 3.0, 0.4, 0.2),
        Token(1, 7.0, 0.6, 0.0),
        Token(1, 7.0, 0.8, 0.2),
    ]


def test_the_latest_of_several_readings_in_one_step_stands() -> None:
    laid = GriddedTokens(2).of([window(timed(1, [(1.0, 0.1), (2.0, 0.3), (5.0, 0.9)]))])[0]

    assert [(token.value, token.time, token.gap) for token in laid] == [
        (2.0, 0.0, 0.0),
        (5.0, 0.5, 0.0),
    ]


def test_static_features_are_left_as_they_are_and_lead_the_window() -> None:
    statics = [Token(9, 1.5, 0.0, 0.0, timeless=True), Token(4, -0.5, 0.0, 0.0, timeless=True)]

    laid = GriddedTokens(4).of([TokenWindow.of([*statics, *timed(2, [(1.0, 0.6)])])])[0]

    assert list(laid) == [
        Token(4, -0.5, 0.0, 0.0, timeless=True),
        Token(9, 1.5, 0.0, 0.0, timeless=True),
        Token(2, 1.0, 0.5, 0.0),
        Token(2, 1.0, 0.75, 0.25),
    ]


def test_a_reading_at_the_window_s_end_falls_in_the_last_step() -> None:
    laid = GriddedTokens(4).of([window(timed(3, [(8.0, 1.0)]))])[0]

    assert list(laid) == [Token(3, 8.0, 0.75, 0.0)]


def test_windows_keep_the_order_they_came_in() -> None:
    first, second = window(timed(1, [(1.0, 0.0)])), window(timed(2, [(2.0, 0.0)]))

    laid = GriddedTokens(1).of([first, second])

    assert [token.channel_id for gridded in laid for token in gridded] == [1, 2]


def test_a_grid_without_steps_is_refused() -> None:
    with pytest.raises(ValueError, match="a grid has steps"):
        GriddedTokens(0)


@st.composite
def irregular_windows(draw: st.DrawFn) -> TokenWindow:
    """A window of up to four channels read at drawn instants, a static feature sometimes."""
    tokens: list[Token] = []
    for channel in draw(st.sets(st.integers(1, 6), min_size=1, max_size=4)):
        times = sorted(draw(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=8)))
        values = draw(st.lists(st.floats(-5.0, 5.0), min_size=len(times), max_size=len(times)))
        tokens.extend(timed(channel, list(zip(values, times, strict=True))))
    if draw(st.booleans()):
        tokens.append(Token(7, draw(st.floats(-5.0, 5.0)), 0.0, 0.0, timeless=True))
    return TokenWindow.of(tokens)


@given(irregular_windows(), st.integers(1, 12))
def test_the_tokens_lay_back_into_the_grid_they_were_read_from(
    raw: TokenWindow, steps: int
) -> None:
    # The token form and the array form of one grid are the same reading: laying the gridded
    # tokens on the grid again gives the values the raw window gives, and a gap of zero marks
    # exactly the steps the raw window observed.
    gridded = GriddedTokens(steps).of([raw])[0]
    grid = RegularGrid(steps, 7)
    from_raw, from_tokens = grid.of([raw])[0], grid.of([gridded])[0]

    assert np.array_equal(from_tokens[:7], from_raw[:7])
    timed_tokens = [token for token in gridded if not token.timeless]
    observed = np.zeros((7, steps))
    for token in timed_tokens:
        if token.gap == 0.0:
            observed[token.channel_id - 1, round(token.time * steps)] = 1.0
    statics = [token.channel_id - 1 for token in raw if token.timeless]
    assert np.array_equal(
        np.delete(observed, statics, axis=0), np.delete(from_raw[7:], statics, axis=0)
    )
    assert all(0.0 <= token.gap <= token.time for token in timed_tokens)
