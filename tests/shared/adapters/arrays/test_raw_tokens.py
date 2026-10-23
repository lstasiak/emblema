import pytest

from emblema.shared.kernel.tokens import N_FEATURES, Token, TokenWindow

np = pytest.importorskip("numpy")

from emblema.shared.adapters.arrays.raw_tokens import RawTokens  # noqa: E402
from emblema.shared.adapters.arrays.token_batch import TokenBatch  # noqa: E402

pytestmark = pytest.mark.ml

WINDOW = TokenWindow.of(
    [Token(3, 1.5, 0.0, 0.0, timeless=True), Token(1, 0.25, 0.5, 0.5), Token(2, -1.0, 0.75, 0.75)]
)
SHORT = TokenWindow.of([Token(2, 0.125, 0.25, 0.25)])


def test_the_raw_layout_is_the_batch_every_encoder_was_trained_under() -> None:
    layout = RawTokens()

    laid = layout.batch(layout.lay([WINDOW, SHORT]), dtype=np.float32)

    assert layout.features == N_FEATURES
    expected = TokenBatch.from_windows([WINDOW, SHORT], dtype=np.float32)
    for name in ("features", "channel_ids", "timestamps", "timeless", "padding_mask"):
        assert np.array_equal(getattr(laid, name), getattr(expected, name)), name
    assert laid.features.dtype == np.float32
    assert laid.windows() == (WINDOW, SHORT)
