import hashlib
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor  # noqa: E402

from emblema.pretraining.adapters.objective.token_masking import TokenMasking  # noqa: E402
from emblema.pretraining.adapters.objective.token_masks import TokenMasks  # noqa: E402
from emblema.pretraining.domain.mask_kind import MaskKind  # noqa: E402
from emblema.pretraining.domain.masking_strategy import MaskingStrategy  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from tests.support.token_tensors import grid_batch, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

CHANNELS, STEPS = 8, 32


def draw(strategy: MaskingStrategy, batch: TokenTensors, *, seed: int = 1) -> TokenMasks:
    return TokenMasking(strategy).draw(batch, torch.Generator().manual_seed(seed))


def fraction(mask: Tensor, over: Tensor) -> float:
    return float((mask & over).sum() / over.sum())


def test_the_draws_hide_the_fraction_the_strategy_expects(strategy: MaskingStrategy) -> None:
    batch = grid_batch(512, CHANNELS, STEPS, seed=1)
    timed = ~batch.timeless

    masks = draw(strategy, batch)

    # A timeless token is out of reach of blocks, so the ratio is stated for the timed ones.
    assert fraction(masks.hidden, timed) == pytest.approx(strategy.expected_ratio, abs=0.02)


def test_each_draw_alone_hides_at_its_own_rate() -> None:
    batch = grid_batch(512, CHANNELS, STEPS, seed=2)
    timed = ~batch.timeless

    whole = draw(
        MaskingStrategy(channel_rate=0.3, block_rate=0, block_span=0.5, token_rate=0), batch
    )
    blocks = draw(
        MaskingStrategy(channel_rate=0, block_rate=0.5, block_span=0.4, token_rate=0), batch
    )
    single = draw(
        MaskingStrategy(channel_rate=0, block_rate=0, block_span=0.5, token_rate=0.2), batch
    )

    assert fraction(whole.hidden, timed) == pytest.approx(0.3, abs=0.02)
    assert fraction(blocks.hidden, timed) == pytest.approx(0.2, abs=0.02)
    assert fraction(single.hidden, timed) == pytest.approx(0.2, abs=0.02)


def test_padding_is_never_hidden(strategy: MaskingStrategy) -> None:
    batch = random_batch(16, 40, seed=3, padding=15)

    masks = draw(strategy, batch)

    assert not (masks.hidden & batch.padding_mask).any()


def test_every_window_keeps_a_visible_token() -> None:
    greedy = MaskingStrategy(channel_rate=0.95, block_rate=0.9, block_span=1.0, token_rate=0.95)
    batch = random_batch(64, 6, seed=4, padding=2)

    masks = draw(greedy, batch)

    visible = ~batch.padding_mask & ~masks.hidden
    assert visible.any(dim=1).all()
    assert masks.hidden.any()


def test_a_window_of_nothing_but_padding_hides_nothing_and_raises_nothing(
    strategy: MaskingStrategy,
) -> None:
    batch = random_batch(2, 8, seed=5)
    batch = TokenTensors(
        batch.features,
        batch.channel_ids,
        batch.timestamps,
        batch.timeless,
        torch.ones_like(batch.padding_mask),
    )

    masks = draw(strategy, batch)

    assert not masks.hidden.any()


def test_a_whole_channel_is_hidden_or_kept_and_never_split() -> None:
    batch = grid_batch(64, CHANNELS, STEPS, seed=6)

    masks = draw(
        MaskingStrategy(channel_rate=0.4, block_rate=0, block_span=0.5, token_rate=0), batch
    )

    for row in range(batch.batch_size):
        for channel in range(1, CHANNELS + 1):
            tokens = batch.channel_ids[row] == channel
            hidden = masks.hidden[row, tokens]
            assert hidden.all() or not hidden.any()


def test_a_block_is_one_span_of_a_channels_time() -> None:
    batch = grid_batch(64, CHANNELS, STEPS, seed=7)

    masks = draw(
        MaskingStrategy(channel_rate=0, block_rate=1.0, block_span=0.25, token_rate=0), batch
    )

    for row in range(batch.batch_size):
        for channel in range(1, CHANNELS + 1):
            tokens = batch.channel_ids[row] == channel
            times = batch.timestamps[row, tokens]
            hidden = masks.hidden[row, tokens]
            assert hidden.any()
            inside = (times >= times[hidden].min()) & (times <= times[hidden].max())
            assert torch.equal(inside, hidden)
            assert float(times[hidden].max() - times[hidden].min()) <= 0.25 + 1e-6


def test_a_block_covers_the_fraction_of_the_window_it_declares() -> None:
    batch = grid_batch(256, CHANNELS, STEPS, seed=8)

    masks = draw(
        MaskingStrategy(channel_rate=0, block_rate=1.0, block_span=0.5, token_rate=0), batch
    )

    assert fraction(masks.hidden, ~batch.timeless) == pytest.approx(0.5, abs=0.03)


def test_a_timeless_token_is_never_inside_a_block() -> None:
    batch = grid_batch(64, CHANNELS, STEPS, seed=9, timeless=3)

    # A block the length of the window swallows a timed channel whole and a timeless token never.
    masks = draw(
        MaskingStrategy(channel_rate=0, block_rate=0.9, block_span=1.0, token_rate=0), batch
    )

    assert not masks.hidden[batch.timeless].any()
    assert fraction(masks.hidden, ~batch.timeless) == pytest.approx(0.9, abs=0.03)


def test_a_timeless_token_can_be_hidden_whole_or_on_its_own() -> None:
    batch = grid_batch(256, CHANNELS, STEPS, seed=10)

    whole = draw(
        MaskingStrategy(channel_rate=0.5, block_rate=0, block_span=0.5, token_rate=0), batch
    )
    single = draw(
        MaskingStrategy(channel_rate=0, block_rate=0, block_span=0.5, token_rate=0.5), batch
    )

    assert fraction(whole.channel, batch.timeless) == pytest.approx(0.5, abs=0.1)
    assert fraction(single.hidden, batch.timeless) == pytest.approx(0.5, abs=0.1)


def test_the_channel_kind_is_read_off_the_outcome_not_the_draw() -> None:
    # Single tokens hidden at a high rate empty some channels of a short window entirely; those
    # tokens are of the channel kind, because nothing of their channel is left to interpolate from.
    batch = grid_batch(256, CHANNELS, 3, seed=11)

    masks = draw(
        MaskingStrategy(channel_rate=0, block_rate=0, block_span=0.5, token_rate=0.7), batch
    )

    assert masks.channel.any()
    for row in range(batch.batch_size):
        for channel in range(1, CHANNELS + 1):
            tokens = batch.channel_ids[row] == channel
            emptied = masks.hidden[row, tokens].all()
            assert masks.channel[row, tokens].all() == emptied


@pytest.mark.parametrize("with_tail", [False, True])
def test_the_kinds_partition_the_hidden_tokens(strategy: MaskingStrategy, with_tail: bool) -> None:
    if with_tail:
        strategy = replace(strategy, horizon_rate=0.5, horizon_min_span=0.15, horizon_max_span=0.5)
    batch = grid_batch(64, CHANNELS, STEPS, seed=12)

    masks = draw(strategy, batch)

    kinds = {kind: masks.of_kind(kind) for kind in MaskKind}
    union = torch.zeros_like(masks.hidden)
    for kind, tokens in kinds.items():
        assert not (union & tokens).any(), kind
        union |= tokens
    assert torch.equal(union, masks.hidden)
    assert {kind for kind, tokens in kinds.items() if tokens.any()} == {
        kind for kind in MaskKind if strategy.draws(kind)
    }


def test_a_strategy_without_a_tail_draws_the_masks_it_drew_before_the_tail_existed(
    strategy: MaskingStrategy,
) -> None:
    digest = hashlib.sha256()
    for batch in (grid_batch(16, 8, 32, seed=21), random_batch(16, 40, seed=22, padding=5)):
        masks = draw(strategy, batch, seed=7)
        for mask in (masks.channel, masks.block, masks.token):
            digest.update(mask.numpy().tobytes())
        assert not masks.horizon.any()

    # Drawn by the code before the tail was added; a change here changes every run's masks.
    assert digest.hexdigest() == (
        "d35c160afc6a6def41e117c017a306b1b92b0b910724784b3e9ecc1ff1e548eb"
    )


def tail_only(*, rate: float = 1.0, shortest: float = 0.2, longest: float = 0.4) -> MaskingStrategy:
    return MaskingStrategy(
        channel_rate=0.0,
        block_rate=0.0,
        block_span=0.5,
        token_rate=0.0,
        horizon_rate=rate,
        horizon_min_span=shortest,
        horizon_max_span=longest,
    )


def test_the_tail_hides_every_channel_from_one_instant_to_the_end_of_the_window() -> None:
    batch = grid_batch(256, CHANNELS, STEPS, seed=31)
    timed = ~batch.timeless

    masks = draw(tail_only(), batch)

    for row in range(batch.batch_size):
        hidden = masks.horizon[row] & timed[row]
        times = batch.timestamps[row]
        start = float(times[hidden].min())
        assert torch.equal(hidden, timed[row] & (times >= start))
        assert 1.0 - 0.4 - 1e-6 <= start <= 1.0 - 0.2 + 1.0 / (STEPS - 1)


def test_the_tail_spans_the_lengths_it_declares_uniformly() -> None:
    batch = grid_batch(2048, CHANNELS, STEPS, seed=32)
    timed = ~batch.timeless

    masks = draw(tail_only(), batch)

    assert fraction(masks.hidden, timed) == pytest.approx(tail_only().expected_ratio, abs=0.02)
    starts = torch.stack(
        [batch.timestamps[row][masks.horizon[row]].min() for row in range(batch.batch_size)]
    )
    assert float(starts.min()) < 0.65
    assert float(starts.max()) > 0.75


def test_a_tail_drawn_at_a_rate_is_lost_by_that_share_of_windows() -> None:
    batch = grid_batch(2048, CHANNELS, STEPS, seed=33)

    masks = draw(tail_only(rate=0.3), batch)

    assert float(masks.horizon.any(dim=1).float().mean()) == pytest.approx(0.3, abs=0.03)


def test_the_tail_never_reaches_a_timeless_token_or_padding() -> None:
    batch = random_batch(64, 40, seed=34, padding=10)

    masks = draw(tail_only(shortest=0.9, longest=0.95), batch)

    assert masks.horizon.any()
    assert not (masks.horizon & batch.timeless).any()
    assert not (masks.horizon & batch.padding_mask).any()


def test_a_token_in_the_tail_is_of_the_tail_unless_its_channel_is_gone() -> None:
    batch = grid_batch(256, CHANNELS, STEPS, seed=35)
    strategy = MaskingStrategy(
        channel_rate=0.2,
        block_rate=0.6,
        block_span=0.5,
        token_rate=0.3,
        horizon_rate=1.0,
        horizon_min_span=0.15,
        horizon_max_span=0.5,
    )

    masks = draw(strategy, batch)

    assert torch.equal(masks.of_kind(MaskKind.HORIZON), masks.horizon & ~masks.channel)
    assert (masks.horizon & masks.token).any()
    assert not (masks.of_kind(MaskKind.TOKEN) & masks.horizon).any()
    assert not (masks.of_kind(MaskKind.BLOCK) & masks.horizon).any()


def test_the_same_seed_draws_the_same_masks(strategy: MaskingStrategy) -> None:
    batch = grid_batch(8, CHANNELS, STEPS, seed=13)

    first, second = draw(strategy, batch, seed=5), draw(strategy, batch, seed=5)
    other = draw(strategy, batch, seed=6)

    assert torch.equal(first.hidden, second.hidden)
    assert torch.equal(first.channel, second.channel)
    assert torch.equal(first.block, second.block)
    assert not torch.equal(first.hidden, other.hidden)


def test_masks_moved_to_a_device_are_the_same_masks(strategy: MaskingStrategy) -> None:
    batch = grid_batch(4, CHANNELS, STEPS, seed=11)
    masks = draw(strategy, batch)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    moved = masks.to(device)
    back = moved.to("cpu")

    assert moved.hidden.device.type == device
    for kind in MaskKind:
        assert torch.equal(back.of_kind(kind), masks.of_kind(kind))
