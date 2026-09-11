from collections.abc import Iterator, Sequence

from torch.utils.data import DataLoader

from emblema.pretraining.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler
from emblema.pretraining.adapters.loaders.window_dataset import WindowDataset
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow


class WindowLoader:
    """Batches of token windows for one epoch of training, in an order the seed decides.

    Asking for batches means naming the epoch, so an order that depends on it cannot be forgotten
    — the failure that gives every epoch the same order is silent, and shows up as a training
    curve nobody can explain. A loader that does not shuffle ignores the epoch it is given.

    Windows within a batch differ in token count, so each batch is padded to its own longest
    window rather than to a corpus-wide maximum: the padding a batch carries is set by the windows
    that happen to meet in it.
    """

    def __init__(
        self,
        windows: Sequence[TokenWindow],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> None:
        """Draw batches of ``batch_size`` windows from ``windows``.

        Args:
            windows: The windows to draw batches from; at least one.
            batch_size: How many windows a batch holds.
            seed: The run's seed, which with the epoch fixes the order.
            shuffle: Whether the order depends on the seed and the epoch at all. A pass that only
                reads, such as validation, keeps the order the windows arrive in.
            drop_last: Whether to leave out a final batch that holds fewer windows.
            num_workers: How many processes collate batches; zero collates in the training process.
                A worker is handed the whole dataset once per epoch, which is cheap for a memory
                map and expensive for windows held as objects.

        Raises:
            ValueError: If no window is given.
        """
        dataset = WindowDataset(windows)
        self._order = SeededShuffleSampler(len(dataset), seed=seed) if shuffle else None
        self._batches = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._order,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=TokenTensors.from_windows,
        )

    def batches_of(self, epoch: int) -> Iterator[TokenTensors]:
        """Every window once, batched, in the order ``epoch`` gives them."""
        if self._order is not None:
            self._order.set_epoch(epoch)
        return iter(self._batches)

    def __len__(self) -> int:
        """How many batches an epoch holds."""
        return len(self._batches)
