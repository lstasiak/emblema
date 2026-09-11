import pytest

from emblema.shared.kernel.tokens import Token, TokenWindow

pytest.importorskip("torch")

from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors

pytestmark = pytest.mark.ml


def window(index: int, tokens: int) -> TokenWindow:
    """A window of ``tokens`` tokens, every one of them carrying ``index`` as its value."""
    return TokenWindow.of(
        Token(channel_id=1 + position, value=float(index), time=position / tokens, gap=0.0)
        for position in range(tokens)
    )


# Lengths chosen so that no two consecutive windows are equally long: a batch is padded to the
# longest window that lands in it, and a corpus of uniform windows would not show that.
LENGTHS = (2, 5, 3, 4, 6, 1, 3)
WINDOWS = [window(index, tokens) for index, tokens in enumerate(LENGTHS, start=1)]


def identities(batch: TokenTensors) -> list[float]:
    """Which windows a batch holds, in the order it holds them."""
    return batch.features[:, 0, 0].tolist()


def epoch_identities(loader: WindowLoader, epoch: int) -> list[float]:
    return [identity for batch in loader.batches_of(epoch) for identity in identities(batch)]


def test_a_batch_is_padded_to_the_longest_window_that_lands_in_it() -> None:
    loader = WindowLoader(WINDOWS, batch_size=2, seed=1, shuffle=False)

    assert [batch.token_count for batch in loader.batches_of(0)] == [5, 4, 6, 3]


def test_a_batch_is_marked_padding_from_the_end_of_each_window() -> None:
    loader = WindowLoader(WINDOWS[:2], batch_size=2, seed=1, shuffle=False)

    first = next(loader.batches_of(0))

    assert first.padding_mask.tolist() == [[False, False, True, True, True], [False] * 5]


def test_an_epoch_delivers_every_window_exactly_once() -> None:
    loader = WindowLoader(WINDOWS, batch_size=2, seed=1)

    assert sorted(epoch_identities(loader, 0)) == [float(index) for index in range(1, 8)]


def test_two_loaders_of_one_seed_deliver_the_windows_in_one_order() -> None:
    first = WindowLoader(WINDOWS, batch_size=2, seed=1)
    second = WindowLoader(WINDOWS, batch_size=2, seed=1)

    assert epoch_identities(first, 4) == epoch_identities(second, 4)


def test_another_seed_delivers_the_windows_in_another_order() -> None:
    first = WindowLoader(WINDOWS, batch_size=2, seed=1)
    second = WindowLoader(WINDOWS, batch_size=2, seed=2)

    assert epoch_identities(first, 0) != epoch_identities(second, 0)


def test_a_later_epoch_delivers_the_windows_in_another_order() -> None:
    loader = WindowLoader(WINDOWS, batch_size=2, seed=1)

    assert epoch_identities(loader, 0) != epoch_identities(loader, 1)


def test_a_loader_that_does_not_shuffle_keeps_the_order_it_was_given() -> None:
    loader = WindowLoader(WINDOWS, batch_size=3, seed=1, shuffle=False)

    assert epoch_identities(loader, 0) == [float(index) for index in range(1, 8)]
    assert epoch_identities(loader, 9) == [float(index) for index in range(1, 8)]


def test_the_windows_left_over_by_the_batch_size_make_a_shorter_batch() -> None:
    loader = WindowLoader(WINDOWS, batch_size=3, seed=1)

    assert [batch.batch_size for batch in loader.batches_of(0)] == [3, 3, 1]
    assert len(loader) == 3


def test_a_short_final_batch_is_dropped_on_request() -> None:
    loader = WindowLoader(WINDOWS, batch_size=3, seed=1, drop_last=True)

    assert [batch.batch_size for batch in loader.batches_of(0)] == [3, 3]
    assert len(loader) == 2


def test_an_epoch_already_being_read_keeps_its_order_when_the_next_is_asked_for() -> None:
    # Asking for an epoch must fix its order there and then. Left to the first step, the order
    # would be whichever epoch was named last — the same silent reordering the seed exists to stop.
    expected = epoch_identities(WindowLoader(WINDOWS, batch_size=2, seed=1), 0)
    loader = WindowLoader(WINDOWS, batch_size=2, seed=1)

    started = loader.batches_of(0)
    loader.batches_of(1)

    assert [identity for batch in started for identity in identities(batch)] == expected


def test_an_epoch_with_no_whole_batch_in_it_delivers_nothing() -> None:
    loader = WindowLoader(WINDOWS[:2], batch_size=3, seed=1, drop_last=True)

    assert list(loader.batches_of(0)) == []
    assert len(loader) == 0


def test_workers_deliver_the_epoch_the_training_process_ordered() -> None:
    # Batches are collated in other processes, but the order is decided in this one; a worker that
    # ordered for itself would make a run depend on how many of them there were.
    in_process = WindowLoader(WINDOWS, batch_size=2, seed=1)
    in_workers = WindowLoader(WINDOWS, batch_size=2, seed=1, num_workers=2)

    assert epoch_identities(in_workers, 3) == epoch_identities(in_process, 3)


def test_a_loader_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        WindowLoader([], batch_size=2, seed=1)
