import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.self_attention import SelfAttention  # noqa: E402

pytestmark = pytest.mark.ml

WIDTH = 16
HEADS = 4


def test_a_window_of_nothing_but_padding_yields_finite_values() -> None:
    attention = SelfAttention(WIDTH, HEADS, dropout=0.0).eval()

    attended = attention(torch.randn(1, 8, WIDTH), torch.ones(1, 8, dtype=torch.bool))

    assert torch.isfinite(attended).all()


def test_padding_positions_are_hidden_from_the_observed_ones() -> None:
    torch.manual_seed(1)
    attention = SelfAttention(WIDTH, HEADS, dropout=0.0).eval()
    states = torch.randn(2, 8, WIDTH)
    padding_mask = torch.zeros(2, 8, dtype=torch.bool)
    padding_mask[:, 5:] = True
    other_under_padding = states.clone()
    other_under_padding[:, 5:] = torch.randn(2, 3, WIDTH)

    torch.testing.assert_close(
        attention(other_under_padding, padding_mask)[:, :5],
        attention(states, padding_mask)[:, :5],
    )
