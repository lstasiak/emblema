import math

import torch
from torch import Tensor, nn

from emblema.evaluation.domain.transfer.lora_spec import LoraSpec


class LoraLinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update added to its output.

    The layer keeps its weights and stops training them; what trains is the product of a
    down-projection to the rank and an up-projection back, scaled by ``alpha / rank``. The
    up-projection starts at zero, so the wrapped layer computes exactly what it did before the
    first step and the pretrained behaviour is the starting point rather than a perturbation of
    it. Dropout, where asked for, acts on the update's input alone (Hu et al., 2022).

    Attributes:
        base: The layer as it was, frozen.
        down: ``[rank, in_features]``, drawn as a linear layer's weight would be.
        up: ``[out_features, rank]``, zero at the start.
    """

    def __init__(self, base: nn.Linear, spec: LoraSpec) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(spec.rank, base.in_features))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.up = nn.Parameter(torch.zeros(base.out_features, spec.rank))
        self.scaling = spec.scaling
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        update = self.dropout(inputs) @ self.down.transpose(0, 1) @ self.up.transpose(0, 1)
        return self.base(inputs) + update * self.scaling
