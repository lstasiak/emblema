import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.evaluation.adapters.torch.clipped_values import ClippedValues  # noqa: E402

pytestmark = pytest.mark.ml


class Echo(nn.Module):
    """Hands back what it was fed, so the test reads what the encoder would read."""

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        return torch.cat([features, channel_ids.unsqueeze(-1).float()], dim=-1)


def test_the_value_is_bounded_and_the_gap_and_the_other_tensors_are_not() -> None:
    features = torch.tensor([[[88.0, 0.9], [-31.0, 0.2], [0.5, 7.0]]])
    channel_ids = torch.tensor([[3, 4, 5]])
    ones = torch.ones(1, 3)

    fed = ClippedValues(Echo(), 5.0)(features, channel_ids, ones, ones.bool(), ones.bool())

    torch.testing.assert_close(fed[..., 0], torch.tensor([[5.0, -5.0, 0.5]]))
    torch.testing.assert_close(fed[..., 1], torch.tensor([[0.9, 0.2, 7.0]]))
    torch.testing.assert_close(fed[..., 2], torch.tensor([[3.0, 4.0, 5.0]]))


def test_the_wrapper_holds_no_weights_of_its_own() -> None:
    inner = nn.Linear(2, 2)

    assert list(ClippedValues(inner, 5.0).parameters()) == list(inner.parameters())
