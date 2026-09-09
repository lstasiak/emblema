"""The input shape of the exported graph, and the variations the export tests feed it."""

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

# Continuous features carried by every token: the observation value and the gap since the previous
# observation in the same channel. The count is fixed at export time; the channel count is not an
# axis of this tensor, it is absorbed into the token count.
N_FEATURES = 2

# Entries in the channel vocabulary the dummy encoder was built with.
N_CHANNELS = 64


@dataclass(frozen=True)
class TokenBatch:
    """A window of tokens as the five tensors the exported graph declares, in that order.

    Attributes:
        features: Continuous per-token features, `[batch, tokens, N_FEATURES]`, float32.
        channel_ids: Entry in the channel vocabulary per token, `[batch, tokens]`, int64.
        timestamps: Time within the window, normalised to `[0, 1]`, `[batch, tokens]`, float32.
        timeless: Marks tokens that carry no time at all, `[batch, tokens]`, bool.
        padding_mask: True where the token is padding, `[batch, tokens]`, bool — the PyTorch
            convention, so the same tensor can reach attention unchanged.
    """

    features: Tensor
    channel_ids: Tensor
    timestamps: Tensor
    timeless: Tensor
    padding_mask: Tensor

    @classmethod
    def random(cls, batch: int, tokens: int, *, seed: int, padding: int = 0) -> Self:
        """A batch of `tokens` tokens per row, the last `padding` of them marked as padding."""
        generator = torch.Generator().manual_seed(seed)
        padding_mask = torch.zeros(batch, tokens, dtype=torch.bool)
        if padding:
            padding_mask[:, -padding:] = True
        # Two timeless tokens per window: static features of the window enter the encoder this way,
        # so every batch the tests build exercises the flag.
        timeless = torch.zeros(batch, tokens, dtype=torch.bool)
        timeless[:, :2] = True
        return cls(
            features=torch.randn(batch, tokens, N_FEATURES, generator=generator),
            channel_ids=torch.randint(0, N_CHANNELS, (batch, tokens), generator=generator),
            timestamps=torch.rand(batch, tokens, generator=generator),
            timeless=timeless,
            padding_mask=padding_mask,
        )

    @property
    def token_count(self) -> int:
        return int(self.padding_mask.shape[1])

    @property
    def args(self) -> tuple[Tensor, ...]:
        """Positional arguments for the eager model and for the exporter."""
        return tuple(getattr(self, field.name) for field in fields(self))

    @property
    def feeds(self) -> dict[str, np.ndarray[Any, Any]]:
        """Named inputs for an ONNX Runtime session."""
        return {field.name: getattr(self, field.name).numpy() for field in fields(self)}

    def permuted(self, *, seed: int) -> Self:
        """The same tokens in a different order — a set carries no order."""
        order = torch.randperm(self.token_count, generator=torch.Generator().manual_seed(seed))
        return self._map(lambda tensor: tensor[:, order])

    def padded_by(self, tokens: int) -> Self:
        """The same observations followed by `tokens` padding tokens."""

        def filler(tensor: Tensor) -> Tensor:
            return tensor.new_zeros((tensor.shape[0], tokens, *tensor.shape[2:]))

        extended = self._map(lambda tensor: torch.cat([tensor, filler(tensor)], dim=1))
        marked = self.padding_mask.new_ones((self.padding_mask.shape[0], tokens))
        return replace(extended, padding_mask=torch.cat([self.padding_mask, marked], dim=1))

    def all_timeless(self) -> Self:
        return replace(self, timeless=torch.ones_like(self.timeless))

    def fully_padded(self) -> Self:
        """A window with no observed token — what an over-filtered request leaves behind."""
        return replace(self, padding_mask=torch.ones_like(self.padding_mask))

    def with_timestamps(self, *, seed: int) -> Self:
        generator = torch.Generator().manual_seed(seed)
        return replace(self, timestamps=torch.rand(self.timestamps.shape, generator=generator))

    def _map(self, transform: Callable[[Tensor], Tensor]) -> Self:
        return replace(self, **{f.name: transform(getattr(self, f.name)) for f in fields(self)})


INPUT_NAMES = tuple(field.name for field in fields(TokenBatch))
