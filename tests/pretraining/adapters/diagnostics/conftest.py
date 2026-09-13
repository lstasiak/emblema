"""Windows whose values are known functions of time, so a baseline's answer is known too."""

from collections.abc import Callable, Mapping, Sequence

import pytest

pytest.importorskip("torch")

import torch

from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import Token, TokenWindow

Signal = Callable[[float], float]


def window(
    channels: Mapping[int, Signal],
    *,
    steps: int,
    timeless: Mapping[int, float] | None = None,
    instants: Sequence[float] | None = None,
) -> TokenWindow:
    """One window: each channel's signal sampled at `steps` even instants, or at `instants`."""
    times = (
        list(instants) if instants is not None else [step / (steps - 1) for step in range(steps)]
    )
    tokens = [
        Token(channel_id=channel, value=signal(time), time=time, gap=0.0)
        for channel, signal in channels.items()
        for time in times
    ]
    tokens += [
        Token(channel_id=channel, value=value, time=0.0, gap=0.0, timeless=True)
        for channel, value in (timeless or {}).items()
    ]
    return TokenWindow.of(tokens)


def batch_of(*windows: TokenWindow) -> TokenTensors:
    return TokenTensors.from_windows(list(windows))


def hiding(
    batch: TokenTensors, *, channels: Sequence[int] = (), where: torch.Tensor | None = None
) -> TokenMasks:
    """Masks hiding whole `channels`, and the positions of `where` as single tokens."""
    nothing = torch.zeros_like(batch.padding_mask)
    whole = (
        torch.isin(batch.channel_ids, torch.tensor(list(channels), dtype=torch.int64))
        if channels
        else nothing
    )
    single = where if where is not None else nothing
    return TokenMasks(
        channel=whole & ~batch.padding_mask, block=nothing, token=single & ~batch.padding_mask
    )
