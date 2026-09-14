from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from emblema.pretraining.domain.mask_kind import MaskKind


@dataclass(frozen=True)
class TokenMasks:
    """Which tokens of a batch are hidden from the encoder, and how each came to be.

    Three boolean tensors of the batch's ``[batch, tokens]`` shape. ``channel`` is read off the
    outcome — a hidden token whose channel keeps no visible token in its window — because that,
    not the draw that hid it, decides what can reconstruct it. ``block`` and ``token`` record the
    draws. A token can be in several of them; ``of_kind`` gives each token one kind, the one whose
    baseline applies: channel first, then block, then token.

    Padding is never hidden: it is not a token. Masks built by hand are held to the same rule by
    whoever scores them, not here.

    Attributes:
        channel: Hidden, and the channel has no visible token left in the window.
        block: Hidden by the block draw of its channel.
        token: Hidden by the draw over single tokens.
    """

    channel: Tensor
    block: Tensor
    token: Tensor

    def to(self, device: torch.device | str) -> Self:
        """The same masks on ``device``, as a batch moved there is read with."""
        return type(self)(
            channel=self.channel.to(device),
            block=self.block.to(device),
            token=self.token.to(device),
        )

    @property
    def hidden(self) -> Tensor:
        """Every hidden token, whatever hid it."""
        return self.channel | self.block | self.token

    def of_kind(self, kind: MaskKind) -> Tensor:
        """The hidden tokens of one kind, each token counted under exactly one kind."""
        if kind is MaskKind.CHANNEL:
            return self.channel
        if kind is MaskKind.BLOCK:
            return self.block & ~self.channel
        return self.token & ~self.channel & ~self.block
