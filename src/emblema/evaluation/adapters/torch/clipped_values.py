import torch
from torch import Tensor, nn


class ClippedValues(nn.Module):
    """An encoder fed every token's value bounded to ``[-bound, bound]``; the gap is left alone.

    Called as the encoder is, with the five tensors of a batch in their order, so a candidate
    holds it where it held the encoder and nothing downstream changes. It holds no weights: the
    bound is the plan's, in the standard deviations the readings are standardised in, and a
    value beyond it reaches the encoder as the bound.

    Attributes:
        encoder: The encoder fed the bounded values.
        bound: The bound either side of zero.
    """

    def __init__(self, encoder: nn.Module, bound: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.bound = bound

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        values = features[..., :1].clamp(-self.bound, self.bound)
        bounded = torch.cat([values, features[..., 1:]], dim=-1)
        states: Tensor = self.encoder(bounded, channel_ids, timestamps, timeless, padding_mask)
        return states
