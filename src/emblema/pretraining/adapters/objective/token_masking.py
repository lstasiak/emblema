import torch
from torch import Tensor

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class TokenMasking:
    """Draws the masks a strategy prescribes over a batch, one draw per window and channel.

    A channel-level draw is one uniform number per window and channel, looked up per token
    through its channel identifier, so a batch of any width and any token count is masked in a
    handful of tensor operations and a channel is hidden whole or not at all. Blocks are spans of
    the window's length, not runs of tokens: how many tokens a block swallows follows from how
    densely the channel is sampled there, which is the fact the model is not allowed to assume.

    Every window keeps at least one visible token. Where the draws would hide them all, the first
    observed token is uncovered — a rare event under any sensible strategy, and one that would
    otherwise ask the model to predict a window from nothing.

    The numbers are drawn where the generator lives and moved to the batch afterwards, so the
    masks a seed yields do not depend on the device the run happens to use.
    """

    def __init__(self, strategy: MaskingStrategy) -> None:
        self.strategy = strategy

    def draw(self, batch: TokenTensors, generator: torch.Generator) -> TokenMasks:
        rows, tokens = batch.padding_mask.shape
        ids = batch.channel_ids
        observed = ~batch.padding_mask
        entries = int(ids.max().item()) + 1

        def per_channel() -> Tensor:
            drawn = torch.rand(rows, entries, generator=generator, device=generator.device)
            return drawn.to(ids.device).gather(1, ids)

        def per_token() -> Tensor:
            drawn = torch.rand(rows, tokens, generator=generator, device=generator.device)
            return drawn.to(ids.device)

        whole = per_channel() < self.strategy.channel_rate
        blocked = per_channel() < self.strategy.block_rate
        start = per_channel() * (1.0 - self.strategy.block_span)
        block = (
            blocked
            & ~batch.timeless
            & (batch.timestamps >= start)
            & (batch.timestamps <= start + self.strategy.block_span)
        )
        token = per_token() < self.strategy.token_rate
        hidden = (whole | block | token) & observed

        visible = observed & ~hidden
        starved = (visible.sum(dim=1) == 0) & observed.any(dim=1)
        if bool(starved.any()):
            uncovered = torch.zeros_like(hidden)
            uncovered[torch.arange(rows, device=ids.device), observed.int().argmax(dim=1)] = starved
            hidden &= ~uncovered
            visible = observed & ~hidden

        visible_per_channel = torch.zeros(
            rows, entries, dtype=torch.int64, device=ids.device
        ).scatter_add(1, ids, visible.long())
        channel = hidden & (visible_per_channel.gather(1, ids) == 0)
        return TokenMasks(channel=channel, block=block & hidden, token=token & hidden)
