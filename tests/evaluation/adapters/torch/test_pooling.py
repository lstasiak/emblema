"""What each pooling of a head computes over a batch, and that a plan's pooling reaches it."""

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.evaluation.adapters.torch.attention_pooling import AttentionPooling  # noqa: E402
from emblema.evaluation.adapters.torch.mean_pooling import MeanPooling  # noqa: E402
from emblema.evaluation.adapters.torch.pooling import pooling_module  # noqa: E402
from emblema.evaluation.adapters.torch.statics_apart_pooling import (  # noqa: E402
    StaticsApartPooling,
)
from emblema.evaluation.adapters.torch.tail_pooling import TailPooling  # noqa: E402
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme  # noqa: E402
from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from scripts.frozen_representations import pooled  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml

WIDTH = 16


def states_of(batch: TokenTensors, *, seed: int = 3) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch.batch_size, batch.token_count, WIDTH, generator=generator)


def pool(module: nn.Module, states: Tensor, batch: TokenTensors) -> Tensor:
    with torch.no_grad():
        pooled_states: Tensor = module(states, batch.padding_mask, batch.timestamps, batch.timeless)
    return pooled_states


def test_the_plan_names_the_module_a_head_pools_with() -> None:
    assert isinstance(pooling_module(HeadPooling.mean(), width=WIDTH), MeanPooling)
    tail = pooling_module(HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2), width=WIDTH)
    assert isinstance(tail, TailPooling)
    assert tail.share == 0.2
    attention = pooling_module(HeadPooling(pooling=PoolingScheme.ATTENTION), width=WIDTH)
    assert isinstance(attention, AttentionPooling)
    assert sum(p.numel() for p in attention.parameters()) == WIDTH


def test_the_mean_is_the_shared_pooling_under_the_heads_calling_convention() -> None:
    batch = random_batch(3, 12, seed=1, padding=4)
    states = states_of(batch)

    torch.testing.assert_close(
        pool(MeanPooling(), states, batch), MaskedMeanPooling()(states, batch.padding_mask)
    )


@pytest.mark.parametrize("share", [0.02, 0.1, 0.2])
def test_the_tail_computes_what_the_diagnostic_script_computed(share: float) -> None:
    batch = random_batch(3, 40, seed=2, padding=6)
    states = states_of(batch)

    from_script = pooled(states, batch, torch.tensor([1]), tails=(share,))

    torch.testing.assert_close(
        pool(TailPooling(share), states, batch), from_script[f"tail_{round(share * 100)}"]
    )


def test_the_tail_keeps_the_late_and_the_timeless_tokens_and_nothing_else() -> None:
    batch = random_batch(1, 6, seed=4)
    batch = TokenTensors(
        batch.features,
        batch.channel_ids,
        torch.tensor([[0.0, 0.0, 0.3, 0.85, 0.9, 0.95]]),
        torch.tensor([[True, True, False, False, False, False]]),
        torch.tensor([[False, False, False, False, False, True]]),
    )
    states = states_of(batch)

    kept = pool(TailPooling(0.1), states, batch)

    # The two static features and the token at 0.9; 0.85 is before the tail and 0.95 is padding.
    torch.testing.assert_close(kept[0], states[0, [0, 1, 4]].mean(dim=0))


def test_a_window_with_nothing_in_its_tail_pools_to_zeros() -> None:
    batch = random_batch(1, 5, seed=5)
    batch = TokenTensors(
        batch.features,
        batch.channel_ids,
        torch.full((1, 5), 0.2),
        torch.zeros(1, 5, dtype=torch.bool),
        batch.padding_mask,
    )

    assert pool(TailPooling(0.1), states_of(batch), batch).abs().sum() == 0.0


def test_attention_starts_as_the_mean_and_leaves_the_padding_out() -> None:
    batch = random_batch(3, 12, seed=6, padding=5)
    states = states_of(batch)
    attention = AttentionPooling(WIDTH)

    torch.testing.assert_close(
        pool(attention, states, batch), MaskedMeanPooling()(states, batch.padding_mask)
    )
    with torch.no_grad():
        attention.query.add_(torch.randn(WIDTH))
    padded = pool(attention, states, batch)
    trimmed = TokenTensors(*(tensor[:, :7] for tensor in batch.args))
    torch.testing.assert_close(padded, pool(attention, states[:, :7], trimmed))


def test_attention_over_nothing_but_padding_stays_finite() -> None:
    batch = random_batch(1, 4, seed=7, padding=4)

    assert torch.isfinite(pool(AttentionPooling(WIDTH), states_of(batch), batch)).all()


@pytest.mark.parametrize("scheme", list(PoolingScheme))
def test_a_plan_that_sets_static_features_apart_wraps_its_scheme(scheme: PoolingScheme) -> None:
    pooling = HeadPooling(pooling=scheme).tuned("statics", "apart")

    module = pooling_module(pooling, width=WIDTH)

    assert isinstance(module, StaticsApartPooling)
    assert isinstance(
        module.readings, type(pooling_module(HeadPooling(pooling=scheme), width=WIDTH))
    )


def test_set_apart_the_readings_and_the_static_features_are_averaged_each_on_their_own() -> None:
    batch = random_batch(1, 6, seed=4)
    batch = TokenTensors(
        batch.features,
        batch.channel_ids,
        torch.tensor([[0.0, 0.0, 0.3, 0.85, 0.9, 0.95]]),
        torch.tensor([[True, True, False, False, False, False]]),
        torch.tensor([[False, False, False, False, False, True]]),
    )
    states = states_of(batch)

    mean = pool(StaticsApartPooling(MeanPooling()), states, batch)
    tail = pool(StaticsApartPooling(TailPooling(0.1)), states, batch)

    statics = states[0, [0, 1]].mean(dim=0)
    torch.testing.assert_close(mean[0], torch.cat((states[0, [2, 3, 4]].mean(dim=0), statics)))
    # The tail no longer keeps the static features: they stand on their own side.
    torch.testing.assert_close(tail[0], torch.cat((states[0, 4], statics)))


def test_attention_set_apart_starts_as_the_two_means() -> None:
    batch = random_batch(3, 12, seed=6, padding=5)
    states = states_of(batch)

    pooled_states = pool(StaticsApartPooling(AttentionPooling(WIDTH)), states, batch)

    torch.testing.assert_close(
        pooled_states, pool(StaticsApartPooling(MeanPooling()), states, batch)
    )


def test_set_apart_a_window_without_static_features_or_readings_gives_zeros_on_that_side() -> None:
    batch = random_batch(2, 8, seed=8)
    no_statics = TokenTensors(
        batch.features,
        batch.channel_ids,
        batch.timestamps,
        torch.zeros_like(batch.timeless),
        batch.padding_mask,
    )
    only_statics = TokenTensors(
        batch.features,
        batch.channel_ids,
        batch.timestamps,
        torch.ones_like(batch.timeless),
        batch.padding_mask,
    )
    states = states_of(batch)

    for scheme in (MeanPooling(), TailPooling(0.2), AttentionPooling(WIDTH)):
        apart = StaticsApartPooling(scheme)
        without = pool(apart, states, no_statics)
        only = pool(apart, states, only_statics)
        assert torch.isfinite(without).all()
        assert torch.isfinite(only).all()
        assert torch.equal(without[:, WIDTH:], torch.zeros(2, WIDTH))
        assert torch.equal(only[:, :WIDTH], torch.zeros(2, WIDTH))


def test_set_apart_the_padding_and_the_order_of_tokens_change_nothing() -> None:
    batch = random_batch(2, 10, seed=9, padding=3)
    states = states_of(batch)
    apart = StaticsApartPooling(MeanPooling())
    order = torch.randperm(batch.token_count, generator=torch.Generator().manual_seed(1))
    shuffled = TokenTensors(*(tensor[:, order] for tensor in batch.args))

    torch.testing.assert_close(pool(apart, states[:, order], shuffled), pool(apart, states, batch))
    trimmed = TokenTensors(*(tensor[:, :7] for tensor in batch.args))
    torch.testing.assert_close(pool(apart, states[:, :7], trimmed), pool(apart, states, batch))
