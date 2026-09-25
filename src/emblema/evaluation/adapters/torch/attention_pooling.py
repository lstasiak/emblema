import math

import torch
from torch import Tensor, nn


class AttentionPooling(nn.Module):
    """A weighted mean of a window's states, the weights read off the states by a learnt query.

    One query for the whole head rather than one per token: each state is scored against it,
    the scores of the observed tokens go through a softmax, and the state is their weighted sum.
    The query starts at zero, so before the first step every token weighs the same and the head
    computes the mean it would have computed without the knob; what it learns from there is
    which part of the window to read from — the end, the moving channels — without being told
    a share. Padding is scored at the lowest value the dtype holds rather than at minus
    infinity, so a window of nothing but padding pools to a finite mean instead of a NaN.

    Attributes:
        query: ``[width]``, zero at the start.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.scale = 1.0 / math.sqrt(width)

    def forward(
        self, states: Tensor, padding_mask: Tensor, timestamps: Tensor, timeless: Tensor
    ) -> Tensor:
        scores = (states @ self.query) * self.scale
        scores = scores.masked_fill(padding_mask, torch.finfo(states.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (states * weights).sum(dim=1)
