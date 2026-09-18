from collections.abc import Iterator, Sequence

from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.ordering import seeded_rank

# What the interleaving ranks under, apart from the order each loader gives its own windows: the
# same seed and epoch rank both, and the two must not be the same draw.
_INTERLEAVING = "interleaving"


class InterleavedLoader:
    """The batches of several loaders in one epoch, taken in turns of one loader each.

    Every loader keeps its own order, so a batch holds windows of one loader and is padded to
    that loader's windows alone; what is drawn here is whose turn it is. A turn is ``group``
    consecutive batches of one loader, so that a caller which folds that many batches into one
    unit — a gradient step, say — folds batches of one loader. The turns are ranked by the seed
    and the epoch, so two runs of one seed interleave the same way on any machine, and a run
    resumed inside an epoch takes the turns it had not taken. What is left of a loader once its
    full turns are taken, fewer batches than a turn, comes after every full turn in loader order:
    at most one short turn a loader an epoch, and none where the group divides the counts. A
    single loader interleaves with nothing and yields its batches as it would alone.
    """

    def __init__(self, loaders: Sequence[WindowLoader], *, seed: int, group: int = 1) -> None:
        """Interleave the batches of ``loaders`` in turns of ``group``, ordered by ``seed``.

        Raises:
            ValueError: If no loader is given, or the group is not positive.
        """
        if not loaders:
            raise ValueError("an interleaving needs at least one loader")
        if group < 1:
            raise ValueError(f"a turn holds at least one batch, got {group}")
        self._loaders = tuple(loaders)
        self._seed = seed
        self._group = group

    def __len__(self) -> int:
        """How many batches an epoch holds, over every loader."""
        return sum(len(loader) for loader in self._loaders)

    def turns_of(self, epoch: int) -> tuple[int, ...]:
        """Which loader each batch of ``epoch`` comes from, in order."""
        counts = [len(loader) for loader in self._loaders]
        full = [owner for owner, count in enumerate(counts) for _ in range(count // self._group)]
        ranked = sorted(
            range(len(full)),
            key=lambda position: seeded_rank(self._seed, _INTERLEAVING, epoch, position),
        )
        taken = [full[position] for position in ranked for _ in range(self._group)]
        left = [owner for owner, count in enumerate(counts) for _ in range(count % self._group)]
        return tuple(taken + left)

    def batches_of(self, epoch: int) -> Iterator[TokenTensors]:
        """Each loader's batches of ``epoch`` once, in the order the turns give them."""
        iterators = [loader.batches_of(epoch) for loader in self._loaders]
        for owner in self.turns_of(epoch):
            yield next(iterators[owner])
