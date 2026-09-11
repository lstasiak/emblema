import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from emblema.shared.kernel.exceptions import InvalidTokenWindowError
from emblema.shared.kernel.tokens import Token, TokenWindow

TIMED = st.builds(
    Token,
    channel_id=st.integers(min_value=1, max_value=5),
    value=st.floats(min_value=-1e6, max_value=1e6),
    time=st.floats(min_value=0.0, max_value=1.0),
    gap=st.just(0.0),
    timeless=st.just(False),
).map(lambda token: Token(token.channel_id, token.value, token.time, token.time * 0.5))
TIMELESS = st.builds(
    Token,
    channel_id=st.integers(min_value=1, max_value=5),
    value=st.floats(min_value=-1e6, max_value=1e6),
    time=st.just(0.0),
    gap=st.just(0.0),
    timeless=st.just(True),
)
# At least one timed token, built in rather than filtered for, so that generation never stalls.
TOKENS = st.tuples(st.lists(TIMED, min_size=1, max_size=8), st.lists(TIMELESS, max_size=4)).map(
    lambda pair: [*pair[0], *pair[1]]
)


def window(*tokens: Token) -> TokenWindow:
    return TokenWindow.of(tokens)


@settings(deadline=None)
@given(tokens=TOKENS, seed=st.randoms(use_true_random=False))
def test_the_same_tokens_in_any_order_make_the_same_window(tokens, seed) -> None:
    shuffled = list(tokens)
    seed.shuffle(shuffled)

    assert TokenWindow.of(shuffled) == TokenWindow.of(tokens)


@given(tokens=TOKENS)
def test_iterating_a_window_gives_its_tokens_back(tokens) -> None:
    assert sorted(TokenWindow.of(tokens), key=repr) == sorted(tokens, key=repr)


def test_timeless_tokens_come_first_then_time_order() -> None:
    late = Token(1, 0.0, 0.8, 0.8)
    early = Token(2, 0.0, 0.2, 0.2)
    static = Token(3, 1.0, 0.0, 0.0, timeless=True)

    assert tuple(window(late, early, static)) == (static, early, late)


def test_ties_in_time_break_on_channel_then_value() -> None:
    tokens = (Token(2, 0.0, 0.5, 0.5), Token(1, 5.0, 0.5, 0.5), Token(1, -5.0, 0.5, 0.5))

    assert tuple(window(*tokens)) == (tokens[2], tokens[1], tokens[0])


def test_the_constructor_rejects_tokens_out_of_canonical_order() -> None:
    with pytest.raises(InvalidTokenWindowError, match="canonical order"):
        TokenWindow((1, 1), (0.0, 0.0), (0.8, 0.2), (0.0, 0.0), (False, False))


def test_columns_must_be_equally_long() -> None:
    with pytest.raises(InvalidTokenWindowError, match="one entry per token"):
        TokenWindow((1, 2), (0.0,), (0.0, 0.0), (0.0, 0.0), (False, False))


def test_a_window_needs_a_timed_token() -> None:
    with pytest.raises(InvalidTokenWindowError, match="timed token"):
        window(Token(1, 0.0, 0.0, 0.0, timeless=True))


def test_an_empty_window_is_rejected() -> None:
    with pytest.raises(InvalidTokenWindowError, match="timed token"):
        window()


@pytest.mark.parametrize("channel_id", [0, -1])
def test_channel_ids_are_positive_because_zero_is_padding(channel_id: int) -> None:
    with pytest.raises(InvalidTokenWindowError, match="positive"):
        window(Token(channel_id, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_values_are_finite(value: float) -> None:
    with pytest.raises(InvalidTokenWindowError, match="finite"):
        window(Token(1, value, 0.0, 0.0))


@pytest.mark.parametrize(("time", "gap"), [(0.5, 0.0), (0.0, 0.5)])
def test_a_timeless_token_carries_no_time_and_no_gap(time: float, gap: float) -> None:
    with pytest.raises(InvalidTokenWindowError, match="timeless"):
        window(Token(1, 0.0, 0.0, 0.0), Token(2, 0.0, time, gap, timeless=True))


@pytest.mark.parametrize("time", [-0.1, 1.1])
def test_time_lies_within_the_window(time: float) -> None:
    with pytest.raises(InvalidTokenWindowError, match="within the window"):
        window(Token(1, 0.0, time, 0.0))


@pytest.mark.parametrize(("time", "gap"), [(0.5, 0.6), (0.5, -0.1)])
def test_gap_never_exceeds_time_nor_drops_below_zero(time: float, gap: float) -> None:
    with pytest.raises(InvalidTokenWindowError, match=r"\[0, time\]"):
        window(Token(1, 0.0, time, gap))


def test_time_may_reach_the_window_end_exactly() -> None:
    assert window(Token(1, 0.0, 1.0, 1.0)).times == (1.0,)


def test_len_counts_tokens_of_both_kinds() -> None:
    assert len(window(Token(1, 0.0, 0.3, 0.3), Token(2, 1.0, 0.0, 0.0, timeless=True))) == 2


def test_the_error_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="timed"):
        window()
