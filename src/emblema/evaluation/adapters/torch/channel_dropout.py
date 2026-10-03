from dataclasses import replace

import torch

from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class ChannelDropout:
    """Withholds whole channels of a window while it is learnt from, each with one probability.

    A window's channel is either read whole or not at all for that step, which is how published
    networks on irregular clinical series perturb their input: the network learns to answer with
    a variable missing, as it will have to on a stay that never had it measured. The tokens
    withheld are marked as padding, so the encoder and the pooling treat them as absent and
    nothing is invented in their place. Static features are never withheld, since those
    networks keep them on a path of their own, and a window is left whole where the draw would
    empty it. The draws come from the generator given, so a run repeats under its seed.

    Attributes:
        rate: Probability a channel of a window is withheld.
    """

    def __init__(self, rate: float, generator: torch.Generator) -> None:
        self.rate = rate
        self._generator = generator

    def applied_to(self, batch: TokenTensors) -> TokenTensors:
        """``batch`` with some channels of each window marked as padding."""
        # The mask is torch's: True where a position is padding, so observed tokens are the rest.
        observed = ~batch.padding_mask
        channels = int(batch.channel_ids.max().item()) + 1
        drawn = torch.rand((batch.batch_size, channels), generator=self._generator)
        withheld = drawn.to(batch.channel_ids.device).gather(1, batch.channel_ids) < self.rate
        withheld &= observed & ~batch.timeless
        emptied = (observed & ~withheld).sum(dim=1) == 0
        withheld &= ~emptied.unsqueeze(1)
        return replace(batch, padding_mask=batch.padding_mask | withheld)
