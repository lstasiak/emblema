from collections.abc import Iterator, Sequence
from itertools import chain

from torch.utils.data import DataLoader

from emblema.shared.adapters.loaders.seeded_shuffle_sampler import SeededShuffleSampler
from emblema.shared.adapters.loaders.window_dataset import WindowDataset
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

    The loader is a shared adapter rather than one context's, because pretraining, evaluation and
    batch inference all feed the same encoder the same way, and a batch built differently for a
    probe than for the run it probes would measure the difference rather than the model.
    """

    def __init__(
        self,
        windows: WindowDataset | Sequence[TokenWindow],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> None:
        """Draw batches of ``batch_size`` windows from ``windows``.

        Args:
            windows: The windows to draw batches from, or a dataset already holding them; at least
                one either way.
            batch_size: How many windows a batch holds.
            seed: The run's seed, which with the epoch fixes the order.
            shuffle: Whether the order depends on the seed and the epoch at all. A pass that only
                reads, such as validation, keeps the order the windows arrive in.
            drop_last: Whether to leave out a final batch that holds fewer windows.
            num_workers: How many processes collate batches; zero collates in the training process.
                A worker is handed the whole dataset when it starts, which is cheap for a memory
                map and expensive for windows held as objects.

        Raises:
            ValueError: If no window is given.
        """
        dataset = windows if isinstance(windows, WindowDataset) else WindowDataset(windows)
        self._order = SeededShuffleSampler(len(dataset), seed=seed) if shuffle else None
        self._batches = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._order,
            drop_last=drop_last,
            num_workers=num_workers,
            # Workers outlive the epoch, because the handover they pay for at startup is the whole
            # cost: a pool started afresh each epoch measured 180 ms a batch against 0.7 ms for one
            # that persists, on windows held as Python objects. Torch rejects the flag without
            # workers to keep alive, so it is conditional rather than constant.
            persistent_workers=num_workers > 0,
            collate_fn=TokenTensors.from_windows,
        )

    def batches_of(self, epoch: int) -> Iterator[TokenTensors]:
        """Each window once, batched, in the order ``epoch`` gives them.

        A window is left out only where ``drop_last`` discards a short final batch.

        The first batch is drawn here rather than on the caller's first step. Drawing it is what
        fixes the order inside the iterator: until a batch is asked for, the iterator has not read
        the epoch, and an epoch requested afterwards would silently reorder one already being read.
        """
        if self._order is not None:
            self._order.set_epoch(epoch)
        batches = iter(self._batches)
        first = next(batches, None)
        return batches if first is None else chain((first,), batches)

    def __len__(self) -> int:
        """How many batches an epoch holds."""
        return len(self._batches)
