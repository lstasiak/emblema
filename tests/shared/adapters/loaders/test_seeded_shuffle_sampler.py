import pytest

pytest.importorskip("torch")

from emblema.shared.adapters.loaders.seeded_shuffle_sampler import (
    SeededShuffleSampler,
)

pytestmark = pytest.mark.ml


def order(size: int, *, seed: int, epoch: int = 0) -> list[int]:
    sampler = SeededShuffleSampler(size, seed=seed)
    sampler.set_epoch(epoch)
    return list(sampler)


def test_the_same_seed_and_epoch_give_the_same_order() -> None:
    assert order(64, seed=1, epoch=3) == order(64, seed=1, epoch=3)


def test_a_later_epoch_reorders_the_positions() -> None:
    assert order(64, seed=1, epoch=0) != order(64, seed=1, epoch=1)


def test_another_seed_reorders_the_positions() -> None:
    assert order(64, seed=1) != order(64, seed=2)


def test_an_epoch_visits_every_position_exactly_once() -> None:
    assert sorted(order(64, seed=1, epoch=7)) == list(range(64))


def test_the_order_is_the_one_the_digest_rule_produces() -> None:
    # Pinned so that a change to the key a position is ranked by cannot pass unnoticed: it would
    # silently give a run recorded before the change a different order when it is replayed.
    assert order(8, seed=1) == [1, 3, 2, 7, 0, 5, 6, 4]
    assert order(8, seed=1, epoch=1) == [6, 2, 3, 1, 4, 7, 0, 5]
    assert order(8, seed=2) == [6, 1, 4, 0, 5, 7, 3, 2]


def test_the_sampler_reports_how_many_positions_it_orders() -> None:
    assert len(SeededShuffleSampler(64, seed=1)) == 64


@pytest.mark.parametrize("size", [0, -1])
def test_a_sampler_needs_a_position_to_order(size: int) -> None:
    with pytest.raises(ValueError, match="at least one position"):
        SeededShuffleSampler(size, seed=1)
