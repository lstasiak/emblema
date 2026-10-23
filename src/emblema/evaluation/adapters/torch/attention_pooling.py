import math

import torch
from torch import Tensor, nn


class AttentionPooling(nn.Module):
    """Weighted means of a window's states, the weights read off the states by learnt queries.

    One query per aspect of the window rather than one per token: each state is scored against
    every query, the scores of the observed tokens go through a softmax per query, and the
    pooled state is the queries' weighted sums laid side by side. Every query starts at zero, so
    before the first step each token weighs the same under each of them and the head computes
    the mean it would have computed without the knob; what the queries learn from there is
    which part of the window to read from — the end, the moving channels, the channel that was
    measured once — without being told a share, and the head's own weights over the slots are
    what tells one query from another as they learn. Padding is scored at the lowest value the
    dtype holds rather than at minus infinity, so a window of nothing but padding pools to a
    finite mean instead of a NaN.

    Attributes:
        query: ``[queries, width]``, zero at the start.
    """

    def __init__(self, width: int, queries: int = 1) -> None:
        """Pool states of ``width`` under ``queries`` learnt queries.

        Raises:
            ValueError: If there is no query.
        """
        super().__init__()
        if queries < 1:
            raise ValueError(f"a pooling has queries, got {queries}")
        self.query = nn.Parameter(torch.zeros(queries, width))
        self.scale = 1.0 / math.sqrt(width)

    def forward(
        self, states: Tensor, padding_mask: Tensor, timestamps: Tensor, timeless: Tensor
    ) -> Tensor:
        # One query at a time, in the arithmetic the single query always used, so a head under
        # one query computes what it computed before there could be several.
        pooled = []
        for query in self.query:
            scores = (states @ query) * self.scale
            scores = scores.masked_fill(padding_mask, torch.finfo(states.dtype).min)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled.append((states * weights).sum(dim=1))
        return torch.cat(pooled, dim=-1)
