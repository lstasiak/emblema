"""The trivial answers to the objective that predicts states, on states built by hand."""

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.diagnostics.latent_interpolation_baseline import (  # noqa: E402
    LatentInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.latent_nearest_baseline import (  # noqa: E402
    LatentNearestBaseline,
)
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.pretraining.adapters.diagnostics.conftest import batch_of, hiding, window  # noqa: E402

pytestmark = pytest.mark.ml

WIDTH = 3


def states_of(batch: TokenTensors) -> Tensor:
    """A state per token that is a known function of its channel and instant: ``[c, t, c·t]``."""
    channel = batch.channel_ids.to(torch.float64)
    time = batch.timestamps.to(torch.float64)
    return torch.stack([channel, time, channel * time], dim=-1)


def test_a_token_hidden_between_visible_neighbours_gets_the_line_between_their_states() -> None:
    batch = batch_of(window({1: lambda t: t, 2: lambda t: 2 * t}, steps=9))
    hidden = (batch.channel_ids == 1) & (batch.timestamps > 0.3) & (batch.timestamps < 0.7)
    masks = hiding(batch, where=hidden)
    states = states_of(batch)

    line = LatentInterpolationBaseline().predict(states, batch, masks)

    # Every coordinate of the state is linear in time within a channel, so the line is exact.
    torch.testing.assert_close(line[hidden], states[hidden])
    assert hidden.sum() == 3


def test_the_nearest_baseline_copies_the_state_of_the_closest_visible_token() -> None:
    batch = batch_of(window({1: lambda t: t}, steps=5))
    hidden = batch.timestamps == 0.5
    masks = hiding(batch, where=hidden)
    states = states_of(batch)

    nearest = LatentNearestBaseline().predict(states, batch, masks)

    # 0.25 and 0.75 are equally near; the earlier wins a tie, as it does for a value.
    torch.testing.assert_close(nearest[hidden], states[batch.timestamps == 0.25])
    assert not torch.allclose(nearest[hidden], states[hidden])


def test_a_hidden_tail_carries_the_last_visible_state_forward() -> None:
    batch = batch_of(window({1: lambda t: t}, steps=5))
    hidden = batch.timestamps > 0.6
    masks = hiding(batch, where=hidden)
    states = states_of(batch)

    line = LatentInterpolationBaseline().predict(states, batch, masks)
    nearest = LatentNearestBaseline().predict(states, batch, masks)

    last = states[batch.timestamps == 0.5].expand(int(hidden.sum()), WIDTH)
    torch.testing.assert_close(line[hidden], last)
    torch.testing.assert_close(nearest[hidden], last)


def test_a_channel_hidden_whole_and_a_timeless_token_get_the_visible_mean_state() -> None:
    batch = batch_of(window({1: lambda t: 3.0, 2: lambda t: t}, steps=4, timeless={3: 1.0}))
    masks = hiding(batch, channels=[1, 3])
    states = states_of(batch)
    visible = ~masks.hidden & ~batch.padding_mask
    mean = states[visible].mean(dim=0)

    for baseline in (LatentInterpolationBaseline(), LatentNearestBaseline()):
        predicted = baseline.predict(states, batch, masks)
        for position in torch.nonzero(masks.hidden[0]).flatten().tolist():
            torch.testing.assert_close(predicted[0, position], mean)


def test_padding_is_neither_a_neighbour_nor_a_target() -> None:
    short = window({1: lambda t: t}, steps=3)
    long = window({1: lambda t: t}, steps=6)
    batch = batch_of(short, long)
    hidden = (batch.timestamps == 0.5) & ~batch.padding_mask
    masks = hiding(batch, where=hidden)
    states = states_of(batch)
    states[0, batch.padding_mask[0]] = 1e6

    line = LatentInterpolationBaseline().predict(states, batch, masks)
    nearest = LatentNearestBaseline().predict(states, batch, masks)

    torch.testing.assert_close(line[hidden], states[hidden])
    assert float(line[0, ~batch.padding_mask[0]].abs().max()) < 10.0
    assert float(nearest[0, ~batch.padding_mask[0]].abs().max()) < 10.0
    assert line.shape == nearest.shape == states.shape
    assert line.dtype == states.dtype


def test_a_hidden_token_s_answer_does_not_leak_into_the_visible_mean_of_the_others() -> None:
    batch = batch_of(window({1: lambda t: t, 2: lambda t: 2 * t}, steps=5))
    hidden = (batch.channel_ids == 1) & (batch.timestamps == 0.5)
    masks = hiding(batch, where=hidden)
    states = states_of(batch)
    mean = states[0][~masks.hidden[0]].mean(dim=0)

    for baseline in (LatentInterpolationBaseline(), LatentNearestBaseline()):
        predicted = baseline.predict(states, batch, masks)
        # Every position that is not the hidden token still answers the visible mean.
        for position in torch.nonzero(~hidden[0]).flatten().tolist():
            torch.testing.assert_close(predicted[0, position], mean)
        assert not torch.allclose(predicted[0][hidden[0]], mean)
