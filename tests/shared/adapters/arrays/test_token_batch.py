import pytest

from emblema.shared.kernel.exceptions import InvalidTokenWindowError
from emblema.shared.kernel.tokens import N_FEATURES, PADDING_CHANNEL_ID, Token, TokenWindow

np = pytest.importorskip("numpy")

from emblema.shared.adapters.arrays.token_batch import TokenBatch  # noqa: E402

pytestmark = pytest.mark.ml

SHORT = TokenWindow.of(
    [Token(3, 1.5, 0.0, 0.0, timeless=True), Token(1, 0.25, 0.5, 0.5), Token(2, -1.0, 0.75, 0.75)]
)
LONG = TokenWindow.of(
    [
        Token(1, 0.1, 0.1, 0.1),
        Token(1, 0.2, 0.4, 0.3),
        Token(2, 0.3, 0.4, 0.4),
        Token(1, 0.4, 0.9, 0.5),
    ]
)


def batch(*windows: TokenWindow, dtype=np.float64) -> TokenBatch:
    return TokenBatch.from_windows(windows, dtype=dtype)


def test_arrays_have_the_shapes_the_encoder_declares() -> None:
    laid_out = batch(SHORT, LONG)

    assert laid_out.features.shape == (2, 4, N_FEATURES)
    for array in (
        laid_out.channel_ids,
        laid_out.timestamps,
        laid_out.timeless,
        laid_out.padding_mask,
    ):
        assert array.shape == (2, 4)
    assert (laid_out.batch_size, laid_out.token_count) == (2, 4)


def test_dtypes_follow_the_convention_of_the_graph() -> None:
    laid_out = batch(SHORT, dtype=np.float32)

    assert laid_out.features.dtype == np.float32
    assert laid_out.timestamps.dtype == np.float32
    assert laid_out.channel_ids.dtype == np.int64
    assert laid_out.timeless.dtype == np.bool_
    assert laid_out.padding_mask.dtype == np.bool_


def test_padding_follows_each_window_and_is_marked_true() -> None:
    laid_out = batch(SHORT, LONG)

    assert laid_out.padding_mask.tolist() == [[False, False, False, True], [False] * 4]


def test_padding_positions_carry_the_padding_id_and_zeros() -> None:
    laid_out = batch(SHORT, LONG)

    assert laid_out.channel_ids[0, 3] == PADDING_CHANNEL_ID
    assert laid_out.features[0, 3].tolist() == [0.0, 0.0]
    assert laid_out.timestamps[0, 3] == 0.0
    assert not laid_out.timeless[0, 3]


def test_tokens_keep_their_canonical_order_in_the_row() -> None:
    laid_out = batch(SHORT)

    assert laid_out.channel_ids[0].tolist() == [3, 1, 2]
    assert laid_out.timeless[0].tolist() == [True, False, False]
    assert laid_out.timestamps[0].tolist() == [0.0, 0.5, 0.75]
    assert laid_out.features[0].tolist() == [[1.5, 0.0], [0.25, 0.5], [-1.0, 0.75]]


def test_a_batch_round_trips_to_its_windows_at_full_precision() -> None:
    assert batch(SHORT, LONG).windows() == (SHORT, LONG)


def test_single_precision_round_trips_within_its_resolution() -> None:
    (decoded,) = batch(LONG, dtype=np.float32).windows()

    assert decoded.channel_ids == LONG.channel_ids
    assert decoded.timeless == LONG.timeless
    assert np.allclose(decoded.values, LONG.values, atol=1e-7)
    assert np.allclose(decoded.times, LONG.times, atol=1e-7)
    assert np.allclose(decoded.gaps, LONG.gaps, atol=1e-7)


def test_permuting_the_tokens_of_a_row_decodes_to_the_same_window() -> None:
    laid_out = batch(LONG)
    order = np.array([2, 0, 3, 1])

    permuted = TokenBatch(
        laid_out.features[:, order],
        laid_out.channel_ids[:, order],
        laid_out.timestamps[:, order],
        laid_out.timeless[:, order],
        laid_out.padding_mask[:, order],
    )

    assert permuted.windows() == (LONG,)


def test_a_row_without_an_observed_token_does_not_decode() -> None:
    laid_out = batch(SHORT)
    all_padding = TokenBatch(
        laid_out.features,
        laid_out.channel_ids,
        laid_out.timestamps,
        laid_out.timeless,
        np.ones_like(laid_out.padding_mask),
    )

    with pytest.raises(InvalidTokenWindowError, match="timed token"):
        all_padding.windows()


def test_a_batch_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        TokenBatch.from_windows([])
