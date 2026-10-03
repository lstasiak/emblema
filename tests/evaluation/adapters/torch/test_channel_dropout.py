import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.channel_dropout import ChannelDropout  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml


def test_a_withheld_channel_goes_whole_and_static_features_and_padding_stay_as_they_were() -> None:
    batch = random_batch(4, 12, seed=3)
    observed = ~batch.padding_mask

    dropped = ChannelDropout(0.5, torch.Generator().manual_seed(1)).applied_to(batch)

    newly_hidden = dropped.padding_mask & observed
    assert (dropped.padding_mask >= batch.padding_mask).all()
    assert newly_hidden.any()
    assert not (newly_hidden & batch.timeless).any()
    for window in range(batch.batch_size):
        for channel in set(batch.channel_ids[window][newly_hidden[window]].tolist()):
            timed = (batch.channel_ids[window] == channel) & observed[window]
            timed &= ~batch.timeless[window]
            assert dropped.padding_mask[window][timed].all()
    assert torch.equal(dropped.features, batch.features)
    assert torch.equal(dropped.channel_ids, batch.channel_ids)


def test_a_window_is_never_emptied_and_a_rate_of_zero_withholds_nothing() -> None:
    batch = random_batch(3, 6, seed=5)

    kept = ChannelDropout(0.0, torch.Generator().manual_seed(1)).applied_to(batch)
    nearly_all = ChannelDropout(0.999, torch.Generator().manual_seed(1)).applied_to(batch)

    assert torch.equal(kept.padding_mask, batch.padding_mask)
    assert ((~nearly_all.padding_mask).sum(dim=1) >= 1).all()


def test_the_draws_repeat_under_the_same_seed() -> None:
    batch = random_batch(4, 12, seed=3)

    first = ChannelDropout(0.5, torch.Generator().manual_seed(9)).applied_to(batch)
    second = ChannelDropout(0.5, torch.Generator().manual_seed(9)).applied_to(batch)

    assert torch.equal(first.padding_mask, second.padding_mask)
