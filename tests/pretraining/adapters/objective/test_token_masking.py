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


def test_the_kinds_partition_the_hidden_tokens(strategy: MaskingStrategy) -> None:
    batch = grid_batch(64, CHANNELS, STEPS, seed=12)

    masks = draw(strategy, batch)

    kinds = [masks.of_kind(kind) for kind in MaskKind]
    assert torch.equal(kinds[0] | kinds[1] | kinds[2], masks.hidden)
    assert not (kinds[0] & kinds[1]).any()
    assert not (kinds[0] & kinds[2]).any()
    assert not (kinds[1] & kinds[2]).any()
    assert all(kind.any() for kind in kinds)


def test_the_same_seed_draws_the_same_masks(strategy: MaskingStrategy) -> None:
    batch = grid_batch(8, CHANNELS, STEPS, seed=13)

    first, second = draw(strategy, batch, seed=5), draw(strategy, batch, seed=5)
    other = draw(strategy, batch, seed=6)

    assert torch.equal(first.hidden, second.hidden)
    assert torch.equal(first.channel, second.channel)
    assert torch.equal(first.block, second.block)
    assert not torch.equal(first.hidden, other.hidden)
