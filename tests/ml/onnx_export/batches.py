"""The input the exported graph is traced and exercised with, and the variations the tests feed it.

A batch is a ``TokenTensors`` — the same value the loader collates windows into. What the exporter
traces has to be what the encoder is really called with, or the suite proves a claim about a shape
nothing produces. Only the apparatus lives here: batches invented out of nothing, and the
deformations the tests apply to them.
"""

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from emblema.shared.adapters.arrays.token_batch import N_FEATURES
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

# Entries in the channel vocabulary the dummy encoder was built with.
N_CHANNELS = 64

# The graph names its inputs after the fields, in the order the model declares them.
INPUT_NAMES = tuple(field.name for field in fields(TokenTensors))


def random_batch(batch: int, tokens: int, *, seed: int, padding: int = 0) -> TokenTensors:
    """A batch of `tokens` tokens per row, the last `padding` of them marked as padding."""
    generator = torch.Generator().manual_seed(seed)
    padding_mask = torch.zeros(batch, tokens, dtype=torch.bool)
    if padding:
        padding_mask[:, -padding:] = True
    # Two timeless tokens per window: static features of the window enter the encoder this way,
    # so every batch the tests build exercises the flag.
    timeless = torch.zeros(batch, tokens, dtype=torch.bool)
    timeless[:, :2] = True
    return TokenTensors(
        features=torch.randn(batch, tokens, N_FEATURES, generator=generator),
        # Identifier 0 is reserved for padding in the representation, so real tokens draw
        # from 1 upwards.
        channel_ids=torch.randint(1, N_CHANNELS, (batch, tokens), generator=generator),
        timestamps=torch.rand(batch, tokens, generator=generator),
        timeless=timeless,
        padding_mask=padding_mask,
    )


def feeds(batch: TokenTensors) -> dict[str, np.ndarray[Any, Any]]:
    """Named inputs for an ONNX Runtime session."""
    return {name: getattr(batch, name).numpy() for name in INPUT_NAMES}


def permuted(batch: TokenTensors, *, seed: int) -> TokenTensors:
    """The same tokens in a different order — a set carries no order."""
    order = torch.randperm(batch.token_count, generator=torch.Generator().manual_seed(seed))
    return _map(batch, lambda tensor: tensor[:, order])


def padded_by(batch: TokenTensors, tokens: int) -> TokenTensors:
    """The same observations followed by `tokens` padding tokens."""

    def filler(tensor: Tensor) -> Tensor:
        return tensor.new_zeros((tensor.shape[0], tokens, *tensor.shape[2:]))

    extended = _map(batch, lambda tensor: torch.cat([tensor, filler(tensor)], dim=1))
    marked = batch.padding_mask.new_ones((batch.padding_mask.shape[0], tokens))
    return replace(extended, padding_mask=torch.cat([batch.padding_mask, marked], dim=1))


def all_timeless(batch: TokenTensors) -> TokenTensors:
    return replace(batch, timeless=torch.ones_like(batch.timeless))


def fully_padded(batch: TokenTensors) -> TokenTensors:
    """A window with no observed token — what an over-filtered request leaves behind."""
    return replace(batch, padding_mask=torch.ones_like(batch.padding_mask))


def with_timestamps(batch: TokenTensors, *, seed: int) -> TokenTensors:
    generator = torch.Generator().manual_seed(seed)
    return replace(batch, timestamps=torch.rand(batch.timestamps.shape, generator=generator))


def _map(batch: TokenTensors, transform: Callable[[Tensor], Tensor]) -> TokenTensors:
    return replace(batch, **{f.name: transform(getattr(batch, f.name)) for f in fields(batch)})
