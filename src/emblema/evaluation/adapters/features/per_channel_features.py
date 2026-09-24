from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.features.window_statistics import WindowStatistics
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.kernel.tokens import TokenWindow


class PerChannelFeatures:
    """Reads a window as one block of statistics per channel of the corpus, in vocabulary order.

    The strongest reading of a single corpus: a tree can learn that it is the fourth channel's
    drift that matters, which a summary across channels can never say. The price is that the
    vector is as wide as the corpus has channels and its columns mean what they mean only in
    that corpus, so a candidate reading windows this way is fitted on one corpus and stays there.

    A channel the window does not hold is not an error — windows of a corpus with static
    features or gaps hold different channels — so its block is a count of zero and nothing else,
    which is exactly what happened.
    """

    def __init__(self, channels: int) -> None:
        """Read windows of a corpus whose vocabulary runs from one to ``channels``.

        Raises:
            ValueError: If the corpus is said to have no channel.
        """
        if channels < 1:
            raise ValueError(f"a corpus has at least one channel, got {channels}")
        self._channels = channels

    @property
    def width(self) -> int:
        return self._channels * len(WindowStatistics.NAMES)

    def names(self) -> tuple[str, ...]:
        """What each column holds, so a stored candidate can be read a year later."""
        return tuple(
            f"channel_{channel}_{name}"
            for channel in range(1, self._channels + 1)
            for name in WindowStatistics.NAMES
        )

    def of(self, windows: Sequence[TokenWindow]) -> NDArray[np.float64]:
        """The windows as one row each, in the order given.

        Raises:
            UnreadableTaskCorpusError: If a window holds a channel the corpus does not name,
                which means the windows and the manifest are not of the same publication.
        """
        rows = np.empty((len(windows), self.width), dtype=np.float64)
        for index, window in enumerate(windows):
            rows[index] = self._row(WindowStatistics.of(window))
        return rows

    def _row(self, statistics: WindowStatistics) -> NDArray[np.float64]:
        highest = int(statistics.channels[-1])
        if highest > self._channels:
            raise UnreadableTaskCorpusError(
                f"a window holds channel {highest} and the corpus names {self._channels}"
            )
        block = np.full((self._channels, len(WindowStatistics.NAMES)), np.nan)
        block[:, 0] = 0.0
        block[statistics.channels - 1] = statistics.rows
        return block.ravel()
