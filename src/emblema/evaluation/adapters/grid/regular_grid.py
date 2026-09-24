from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.kernel.tokens import TokenWindow


class RegularGrid:
    """Lays a window of tokens on equal steps, for a method that can read nothing else.

    A token's time is a fraction of its window, so the grid divides the window into equal steps
    and a reading falls into the step that holds its instant; where several do, the latest one
    stands. A step with no reading carries the channel's last one forward, and a channel not yet
    observed carries zero — the corpus mean, since values are normalised — until it is. A static
    feature holds its value at every step.

    Carrying readings forward invents values the sensor never reported, which is exactly what a
    resampled irregular corpus is and what a method on a grid has to pay. So every channel is
    doubled by a mask of the steps where it was really observed, and the method sees both what
    the grid says and how much of it was made up.

    The shape is ``[windows, 2 × channels, steps]``: the values of channels one to ``channels``
    in vocabulary order, then their masks in the same order. The channels are an axis here, and
    that is why nothing laid on this grid can leave its corpus.
    """

    def __init__(self, steps: int, channels: int) -> None:
        """Lay windows of a corpus of ``channels`` channels on ``steps`` equal steps.

        Raises:
            ValueError: If there is no step or no channel.
        """
        if steps < 1 or channels < 1:
            raise ValueError(f"a grid has steps and channels, got {steps} and {channels}")
        self._steps = steps
        self._channels = channels

    def of(self, windows: Sequence[TokenWindow]) -> NDArray[np.float64]:
        """The windows on the grid, one slab each, in the order given.

        Raises:
            UnreadableTaskCorpusError: If a window holds a channel the corpus does not name.
        """
        laid = np.zeros((len(windows), 2 * self._channels, self._steps), dtype=np.float64)
        for index, window in enumerate(windows):
            values, observed = self._laid(window)
            laid[index, : self._channels] = values
            laid[index, self._channels :] = observed
        return laid

    def _laid(self, window: TokenWindow) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ids = np.asarray(window.channel_ids, dtype=np.int64) - 1
        if int(ids.max()) >= self._channels:
            raise UnreadableTaskCorpusError(
                f"a window holds channel {int(ids.max()) + 1} and the corpus names {self._channels}"
            )
        values = np.asarray(window.values, dtype=np.float64)
        timeless = np.asarray(window.timeless, dtype=bool)
        steps = np.minimum(
            (np.asarray(window.times, dtype=np.float64) * self._steps).astype(np.int64),
            self._steps - 1,
        )
        grid = np.zeros((self._channels, self._steps), dtype=np.float64)
        observed = np.zeros((self._channels, self._steps), dtype=bool)
        # The latest reading of a step stands: the window is in time order, so keep the last
        # occurrence of every cell rather than rely on the order fancy assignment writes in.
        timed = ~timeless
        cells = ids[timed] * self._steps + steps[timed]
        reversed_first = np.unique(cells[::-1], return_index=True)[1]
        last = len(cells) - 1 - reversed_first
        grid.flat[cells[last]] = values[timed][last]
        observed.flat[cells[last]] = True
        latest = np.maximum.accumulate(np.where(observed, np.arange(self._steps), -1), axis=1)
        filled = np.where(latest >= 0, np.take_along_axis(grid, np.maximum(latest, 0), axis=1), 0.0)
        filled[ids[timeless]] = values[timeless][:, None]
        observed[ids[timeless]] = True
        return filled, observed.astype(np.float64)
