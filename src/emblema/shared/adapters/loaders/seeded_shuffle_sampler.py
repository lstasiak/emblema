from collections.abc import Iterator

from torch.utils.data import Sampler

from emblema.shared.kernel.ordering import seeded_rank


class SeededShuffleSampler(Sampler[int]):
    """The order a seed and an epoch give to a fixed number of windows.

    Each position is ranked by ``seeded_rank`` of itself under the seed and the epoch, and the
    order is that ranking. Two runs of the same seed see the same windows in the same order in
    every epoch, on any machine and under any version of torch, numpy or Python — none of their
    generators is involved. That matters because a run is prepared on one machine and executed on
    another.

    The epoch is state of the sampler rather than of iteration, because a run resumed from a
    checkpoint must continue the sequence rather than restart it; ``WindowLoader`` makes naming it
    part of asking for an epoch's batches. Iteration takes its own copy of the order, so an epoch
    already being read is unaffected by the next one being set.
    """

    def __init__(self, size: int, *, seed: int, turn: int = 0) -> None:
        """Order ``size`` positions under ``seed``.

        Args:
            size: How many positions to order; positive.
            seed: The run's seed.
            turn: Which pass over the same positions within one epoch this is, where a run reads
                them more than once an epoch; each turn is an order of its own. The first turn
                ranks as every run before turns existed, so their orders replay unchanged.

        Raises:
            ValueError: If ``size`` is not positive, or the turn is negative.
        """
        if size <= 0:
            raise ValueError(f"a sampler needs at least one position, got {size}")
        if turn < 0:
            raise ValueError(f"a turn counts from zero, got {turn}")
        self._size = size
        self._seed = seed
        self._turn = turn
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Order the positions as they are ordered in ``epoch``."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(range(self._size), key=self._rank))

    def _rank(self, position: int) -> bytes:
        if self._turn == 0:
            return seeded_rank(self._seed, self._epoch, position)
        return seeded_rank(self._seed, self._epoch, position, "turn", self._turn)
