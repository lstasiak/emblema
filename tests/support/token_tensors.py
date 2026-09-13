"""Batches invented out of nothing, and the deformations tests apply to them.

A batch is a ``TokenTensors`` — the same value the loader collates windows into — so what the
encoder and the exporter are exercised with is what they are really called with. Only apparatus
lives here: random batches of the right shape and types, and the ways of bending one that the
invariance tests need.
"""

from collections.abc import Callable
from dataclasses import fields, replace

import torch
from torch import Tensor

from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import N_FEATURES

# Entries in the channel vocabulary the test encoders are built over.
VOCABULARY_SIZE = 64


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
        channel_ids=torch.randint(1, VOCABULARY_SIZE + 1, (batch, tokens), generator=generator),
        timestamps=torch.rand(batch, tokens, generator=generator),
        timeless=timeless,
        padding_mask=padding_mask,
    )


def grid_batch(
    windows: int, channels: int, steps: int, *, seed: int, timeless: int = 1
) -> TokenTensors:
    """Windows of `channels` channels each observed at the same `steps` evenly spaced instants.

    Values are random; the layout is regular so that a test can count what a draw over channels
    and spans of time should hit. Channels draw identifiers from 1 upwards, the `timeless` static
    features of each window continue from there and carry time zero.
    """
    generator = torch.Generator().manual_seed(seed)
    tokens = channels * steps + timeless
    times = torch.linspace(0.0, 1.0, steps)
    timestamps = torch.cat([torch.zeros(timeless), times.repeat_interleave(channels)])
    channel_ids = torch.cat(
        [
            torch.arange(channels + 1, channels + 1 + timeless),
            torch.arange(1, channels + 1).repeat(steps),
        ]
    )
    flags = torch.cat(
        [torch.ones(timeless, dtype=torch.bool), torch.zeros(channels * steps, dtype=torch.bool)]
    )
    return TokenTensors(
        features=torch.randn(windows, tokens, N_FEATURES, generator=generator),
        channel_ids=channel_ids.expand(windows, tokens).clone(),
        timestamps=timestamps.expand(windows, tokens).clone(),
        timeless=flags.expand(windows, tokens).clone(),
        padding_mask=torch.zeros(windows, tokens, dtype=torch.bool),
    )


def permutation(batch: TokenTensors, *, seed: int) -> Tensor:
    """An order to visit the tokens of `batch` in, the same for every row."""
    return torch.randperm(batch.token_count, generator=torch.Generator().manual_seed(seed))


def permuted(batch: TokenTensors, *, seed: int) -> TokenTensors:
    """The same tokens in a different order — a set carries no order."""
    order = permutation(batch, seed=seed)
    return _map(batch, lambda tensor: tensor[:, order])


def padded_by(batch: TokenTensors, tokens: int) -> TokenTensors:
    """The same observations followed by `tokens` padding tokens."""

    def filler(tensor: Tensor) -> Tensor:
        return tensor.new_zeros((tensor.shape[0], tokens, *tensor.shape[2:]))

    extended = _map(batch, lambda tensor: torch.cat([tensor, filler(tensor)], dim=1))
    marked = batch.padding_mask.new_ones((batch.padding_mask.shape[0], tokens))
    return replace(extended, padding_mask=torch.cat([batch.padding_mask, marked], dim=1))


def scrambled_under_padding(batch: TokenTensors, *, seed: int) -> TokenTensors:
    """The same batch with other values, channels, times and flags at its padding positions.

    A padding position carries zeros by convention; a model that is indifferent to padding must
    be indifferent to what the position carries, not merely to zeros.
    """
    other = random_batch(batch.batch_size, batch.token_count, seed=seed)
    hidden = batch.padding_mask
    return replace(
        batch,
        features=torch.where(hidden.unsqueeze(-1), other.features, batch.features),
        channel_ids=torch.where(hidden, other.channel_ids, batch.channel_ids),
        timestamps=torch.where(hidden, other.timestamps, batch.timestamps),
        # A padding position carries the flag too, and `random_batch` leaves it false there, so
        # the deformation is to raise it rather than to draw it again.
        timeless=batch.timeless | hidden,
    )


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
