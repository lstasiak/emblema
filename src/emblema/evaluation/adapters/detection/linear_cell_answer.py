from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.hidden_cells import HiddenCells


@dataclass(frozen=True)
class LinearCellAnswer:
    """A ridge per channel filling in a hidden cell's five values from the cells left visible.

    A hidden cell is described by what an answer that learnt nothing from the backbone may read
    of the same draw: its channel's line between the visible values on either side and its last
    visible value, each with a mark of whether there was one; every other channel's visible mean
    in the same bin with a mark of whether it held a reading; and the bin. Each channel has its
    own ridge, fitted on the hidden cells of the fitting side, its intercept unpenalised and its
    penalty chosen by the squared error on the other half of the fitting units. A channel the
    fitting side never hid a held cell of is answered by its line alone.

    Attributes:
        channels: The channels a column of features stands for, in order.
        bins: Spans a window's channel is laid over.
        weights: ``{channel: [features, 5]}`` — the ridge of every channel fitted.
    """

    PENALTIES: ClassVar[tuple[float, ...]] = (1.0, 10.0, 100.0, 1000.0)
    # Cells described before the sums of a half take them in.
    _FLUSH_ROWS: ClassVar[int] = 65536

    channels: tuple[int, ...]
    bins: int
    weights: dict[int, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def features(self) -> int:
        return 5 + 2 * len(self.channels) + self.bins

    def described(self, cells: HiddenCells) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """``[scored, features]`` — every scored cell described, beside its channel.

        Raises:
            ValueError: If a window holds a channel the answer was not laid out over.
        """
        index = {channel: at for at, channel in enumerate(self.channels)}
        unknown = [int(c) for c in cells.channels if int(c) not in index]
        if unknown:
            raise ValueError(f"a window holds the channel {unknown[0]}, which no fit laid out")
        columns = np.asarray([index[int(c)] for c in cells.channels], dtype=np.int64)
        bins = cells.scored.shape[0]
        others = np.zeros((bins, 2 * len(self.channels)))
        others[:, columns] = cells.visible_means
        others[:, len(self.channels) + columns] = cells.visible_presence
        at_bin, at_column = np.nonzero(cells.scored)
        described = np.empty((len(at_bin), self.features))
        described[:, 0] = 1.0
        described[:, 1] = cells.interpolated[at_bin, at_column]
        described[:, 2] = cells.either[at_bin, at_column]
        described[:, 3] = cells.carried[at_bin, at_column]
        described[:, 4] = cells.before[at_bin, at_column]
        block = others[at_bin]
        # A cell's own channel is hidden there, so its column says nothing it has not said above.
        own = columns[at_column]
        block[np.arange(len(own)), own] = 0.0
        block[np.arange(len(own)), len(self.channels) + own] = 0.0
        described[:, 5 : 5 + 2 * len(self.channels)] = block
        described[:, 5 + 2 * len(self.channels) :] = np.eye(self.bins)[at_bin]
        return described, cells.channels[at_column]

    def answers(self, cells: HiddenCells) -> NDArray[np.float64]:
        """``[bins, channels, 5]`` — the five values of every scored cell; the line elsewhere."""
        answered = cells.interpolation()
        described, channels = self.described(cells)
        at_bin, at_column = np.nonzero(cells.scored)
        for channel, weights in self.weights.items():
            rows = channels == channel
            if rows.any():
                answered[at_bin[rows], at_column[rows]] = described[rows] @ weights
        return answered

    @classmethod
    def fitted(
        cls,
        channels: Sequence[int],
        bins: int,
        draws: Iterable[tuple[HiddenCells, int]],
    ) -> Self:
        """The ridges fitted on ``draws``, each with the half of the fitting units it came from.

        The penalty of a channel is the one whose ridges, fitted on either half, miss the other
        half's cells least in squared error; the ridge kept is then fitted on both halves.
        """
        answer = cls(tuple(sorted(int(c) for c in channels)), bins)
        by_channel: list[dict[int, _Statistics]] = [{}, {}]
        pending: list[list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]]]
        pending = [[], []]
        held = [0, 0]

        def flush(half: int) -> None:
            # One update per channel for many draws: the sums cost a product per channel, and
            # a draw holds only a few cells of each.
            if not pending[half]:
                return
            described = np.concatenate([rows for rows, _, _ in pending[half]])
            targets = np.concatenate([answers for _, answers, _ in pending[half]])
            of_channel = np.concatenate([channels for _, _, channels in pending[half]])
            for channel in np.unique(of_channel):
                rows = of_channel == channel
                per_half = by_channel[half].setdefault(int(channel), _Statistics(answer.features))
                per_half.add(described[rows], targets[rows])
            pending[half].clear()
            held[half] = 0

        for cells, half in draws:
            described, of_channel = answer.described(cells)
            pending[half].append((described, cells.targets[cells.scored], of_channel))
            held[half] += len(described)
            if held[half] >= cls._FLUSH_ROWS:
                flush(half)
        flush(0)
        flush(1)
        weights = {}
        for channel in answer.channels:
            halves = [by_channel[0].get(channel), by_channel[1].get(channel)]
            if all(half is None for half in halves):
                continue
            both = _Statistics.summed(answer.features, halves)
            penalty = min(
                cls.PENALTIES,
                key=lambda penalty: sum(
                    held.missed(fit.solved(penalty))
                    for fit, held in ((halves[0], halves[1]), (halves[1], halves[0]))
                    if fit is not None and held is not None
                ),
            )
            weights[channel] = both.solved(penalty)
        return cls(answer.channels, bins, weights)


@dataclass
class _Statistics:
    """The sums a ridge is solved from and a fit's squared error on these rows is read from."""

    features: int
    gram: NDArray[np.float64] = field(init=False)
    cross: NDArray[np.float64] = field(init=False)
    squares: float = 0.0

    def __post_init__(self) -> None:
        self.gram = np.zeros((self.features, self.features))
        self.cross = np.zeros((self.features, 5))

    def add(self, described: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        self.gram += described.T @ described
        self.cross += described.T @ targets
        self.squares += float((targets * targets).sum())

    @classmethod
    def summed(cls, features: int, parts: Sequence["_Statistics | None"]) -> "_Statistics":
        total = cls(features)
        for part in parts:
            if part is not None:
                total.gram += part.gram
                total.cross += part.cross
                total.squares += part.squares
        return total

    def solved(self, penalty: float) -> NDArray[np.float64]:
        """The ridge on these rows, its intercept (the first feature) unpenalised."""
        diagonal = np.full(self.features, penalty)
        # The intercept takes no penalty; a whisker keeps a channel never seen solvable.
        diagonal[0] = 1e-9
        solved: NDArray[np.float64] = np.linalg.solve(self.gram + np.diag(diagonal), self.cross)
        return solved

    def missed(self, weights: NDArray[np.float64]) -> float:
        """The squared error of ``weights`` over these rows, from their sums alone."""
        return float(
            self.squares
            - 2.0 * np.sum(weights * self.cross)
            + np.sum(weights * (self.gram @ weights))
        )
