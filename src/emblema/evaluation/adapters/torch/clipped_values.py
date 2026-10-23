import torch
from torch import Tensor, nn


class ClippedValues(nn.Module):
    """An encoder fed every token's values bounded to ``[-bound, bound]``; the rest is left alone.

    Called as the encoder is, with the five tensors of a batch in their order, so a candidate
    holds it where it held the encoder and nothing downstream changes. It holds no weights: the
    bound is the plan's, in the standard deviations the readings are standardised in, and a
    value beyond it reaches the encoder as the bound. Which features are values is the
    layout's to say: the first of a raw token, the five levels of a cell; the gap, the presence
    and the counts are never bounded.

    Attributes:
        encoder: The encoder fed the bounded values.
        bound: The bound either side of zero.
        values: Which features of a token hold its values.
    """

    def __init__(self, encoder: nn.Module, bound: float, values: slice = slice(0, 1)) -> None:
        super().__init__()
        self.encoder = encoder
        self.bound = bound
        self.values = values

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        start, stop, _ = self.values.indices(features.shape[-1])
        bounded = torch.cat(
            [
                features[..., :start],
                features[..., start:stop].clamp(-self.bound, self.bound),
                features[..., stop:],
            ],
            dim=-1,
        )
        states: Tensor = self.encoder(bounded, channel_ids, timestamps, timeless, padding_mask)
        return states
