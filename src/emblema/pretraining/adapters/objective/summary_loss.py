from dataclasses import dataclass

import torch
from torch import Tensor

from emblema.pretraining.adapters.objective.reconstruction_loss import elementwise, losses_dtype
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.domain.training.summary_term import SUMMARIES
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


@dataclass(frozen=True)
class SummaryUnits:
    """Each channel of each window that lost an observed token with a time: one unit of the term.

    A timeless token is a channel's one value for the whole window, so it has no hidden part to
    summarise and joins no unit; padding is never a token.

    Attributes:
        unit: ``[batch, tokens]`` int64, the unit each scored token belongs to, ``-1`` elsewhere.
        count: How many units the batch holds.
    """

    unit: Tensor
    count: int

    @classmethod
    def of(cls, batch: TokenTensors, hidden: Tensor) -> "SummaryUnits":
        scored = hidden & ~batch.padding_mask & ~batch.timeless
        channels = int(batch.channel_ids.max().item()) + 1 if batch.channel_ids.numel() else 1
        rows = torch.arange(batch.batch_size, device=hidden.device).unsqueeze(1)
        keys = rows * channels + batch.channel_ids
        found, inverse = torch.unique(keys[scored], return_inverse=True)
        unit = torch.full_like(batch.channel_ids, -1)
        unit[scored] = inverse
        return cls(unit=unit, count=int(found.numel()))

    def pooled(self, states: Tensor) -> Tensor:
        """The mean of ``states`` over each unit's tokens, ``[units, width]``."""
        scored = self.unit >= 0
        width = states.shape[-1]
        summed = states.new_zeros(self.count, width).index_add(0, self.unit[scored], states[scored])
        return summed / self.sizes().to(states.dtype).unsqueeze(-1)

    def sizes(self) -> Tensor:
        """The tokens each unit holds, counted by an index sum the accelerator also runs."""
        index = self.unit[self.unit >= 0]
        return torch.zeros(self.count, dtype=torch.int64, device=index.device).index_add(
            0, index, torch.ones_like(index)
        )

    def summaries(self, values: Tensor) -> Tensor:
        """The mean, minimum and maximum of ``values`` over each unit's tokens, ``[units, 3]``."""
        scored = self.unit >= 0
        index, kept = self.unit[scored], values[scored]
        mean = values.new_zeros(self.count).index_add(0, index, kept) / self.sizes().to(
            values.dtype
        )
        lowest = values.new_zeros(self.count).scatter_reduce(
            0, index, kept, reduce="amin", include_self=False
        )
        highest = values.new_zeros(self.count).scatter_reduce(
            0, index, kept, reduce="amax", include_self=False
        )
        return torch.stack([mean, lowest, highest], dim=-1)


class SummaryLoss:
    """Each unit's summaries against what the head answered, a unit weighing one however large.

    The miss is read as the run reads a token's — squared or bounded — summary by summary, and a
    unit costs the mean over its summaries. The summaries are of the normalised values, so the
    trivial answer, a channel's mean, is zero for each of them, as it is for a token.
    """

    def __init__(self, loss: ObjectiveLoss) -> None:
        self.loss = loss

    def summed(
        self, answered: Tensor, batch: TokenTensors, units: SummaryUnits
    ) -> tuple[Tensor, Tensor]:
        """The units' losses before they are divided, and how many units were scored."""
        dtype = losses_dtype(answered)
        target = units.summaries(batch.features[..., 0].to(dtype))
        per_unit = elementwise(self.loss, answered.to(dtype), target).mean(dim=-1)
        return per_unit.sum(), torch.tensor(units.count, device=answered.device)

    def summed_trivial(self, batch: TokenTensors, units: SummaryUnits) -> Tensor:
        """What answering every summary with the channel's mean costs the units."""
        answered = torch.zeros(units.count, len(SUMMARIES), device=batch.features.device)
        trivial, _ = self.summed(answered, batch, units)
        return trivial
