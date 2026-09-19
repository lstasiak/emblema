from collections import Counter

import pytest

pytest.importorskip("torch")

import torch

from emblema.shared.adapters.loaders.interleaved_loader import InterleavedLoader
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from tests.support.experiments import windows

pytestmark = pytest.mark.ml


def loader(count: int, *, seed: int = 1, batch_size: int = 2) -> WindowLoader:
    return WindowLoader(windows(count, seed=seed), batch_size=batch_size, seed=seed)


def test_every_batch_of_every_loader_comes_once() -> None:
    interleaved = InterleavedLoader([loader(8), loader(4, seed=2)], seed=1)

    turns = interleaved.turns_of(0)
    batches = list(interleaved.batches_of(0))

    assert len(interleaved) == len(turns) == len(batches) == 6
    assert Counter(turns) == {0: 4, 1: 2}
    assert all(batch.batch_size == 2 for batch in batches)


def test_the_turns_are_the_same_for_one_seed_and_epoch_and_differ_across_epochs() -> None:
    first = InterleavedLoader([loader(8), loader(8, seed=2)], seed=1)
    second = InterleavedLoader([loader(8), loader(8, seed=2)], seed=1)

    assert first.turns_of(0) == second.turns_of(0)
    assert first.turns_of(0) != first.turns_of(1)
    assert Counter(first.turns_of(1)) == {0: 4, 1: 4}


def test_the_turns_depend_on_the_seed() -> None:
    assert InterleavedLoader([loader(8), loader(8, seed=2)], seed=1).turns_of(
        0
    ) != InterleavedLoader([loader(8), loader(8, seed=2)], seed=2).turns_of(0)


def test_a_turn_is_a_group_of_consecutive_batches_of_one_loader() -> None:
    interleaved = InterleavedLoader([loader(8), loader(4, seed=2)], seed=1, group=2)

    turns = interleaved.turns_of(0)

    assert Counter(turns) == {0: 4, 1: 2}
    assert all(turns[at] == turns[at + 1] for at in range(0, len(turns), 2))
    assert turns != InterleavedLoader([loader(8), loader(4, seed=2)], seed=1).turns_of(0)


def test_what_is_left_of_a_loader_after_its_full_turns_comes_last() -> None:
    # Eight, six and two batches: two full turns, one, and none, with two batches over each of
    # the last two loaders.
    interleaved = InterleavedLoader(
        [loader(16), loader(12, seed=2), loader(4, seed=3)], seed=1, group=4
    )

    turns = interleaved.turns_of(0)

    assert len(turns) == 16
    assert Counter(turns[:12]) == {0: 8, 1: 4}
    assert all(len(set(turns[at : at + 4])) == 1 for at in range(0, 12, 4))
    assert turns[12:] == (1, 1, 2, 2)


def test_a_single_loader_yields_its_batches_as_it_would_alone() -> None:
    alone = loader(8)

    for group in (1, 3):
        interleaved = InterleavedLoader([loader(8)], seed=1, group=group)
        assert interleaved.turns_of(3) == (0, 0, 0, 0)
        for batch, expected in zip(interleaved.batches_of(3), alone.batches_of(3), strict=True):
            assert torch.equal(batch.channel_ids, expected.channel_ids)
            assert torch.equal(batch.features, expected.features)


def test_each_loader_keeps_its_own_order_whoever_s_turn_it_is() -> None:
    own = loader(8, seed=2)
    interleaved = InterleavedLoader([loader(8), loader(8, seed=2)], seed=1)

    from_second = [
        batch
        for owner, batch in zip(interleaved.turns_of(0), interleaved.batches_of(0), strict=True)
        if owner == 1
    ]
    for batch, expected in zip(from_second, own.batches_of(0), strict=True):
        assert torch.equal(batch.features, expected.features)


def test_no_loader_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one loader"):
        InterleavedLoader([], seed=1)


def test_an_empty_turn_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        InterleavedLoader([loader(8)], seed=1, group=0)
