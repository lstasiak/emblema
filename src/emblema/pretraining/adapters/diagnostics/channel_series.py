from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChannelSeries:
    """The visible tokens of one channel in one window, in time order.

    What a trivial baseline works from: the values a channel shows on either side of a hidden
    instant, or at the instant nearest to one. Built per window from the batch's arrays by
    ``of_window``, once for every channel that has a visible token.

    Attributes:
        times: Instants of the visible tokens, increasing.
        values: Their normalised values.
    """

    times: NDArray[np.float64]
    values: NDArray[np.float64]

    @classmethod
    def of_window(
        cls,
        channel_ids: NDArray[np.int64],
        times: NDArray[np.float64],
        values: NDArray[np.float64],
        visible: NDArray[np.bool_],
    ) -> Mapping[int, "ChannelSeries"]:
        """The series of every channel with a visible token in the window, by channel identifier."""
        series: dict[int, ChannelSeries] = {}
        for channel in np.unique(channel_ids[visible]):
            tokens = visible & (channel_ids == channel)
            order = np.argsort(times[tokens], kind="stable")
            series[int(channel)] = cls(times[tokens][order], values[tokens][order])
        return series

    def interpolate(self, at: NDArray[np.float64]) -> NDArray[np.float64]:
        """The value at ``at`` linearly between the visible tokens on either side.

        Past the first or last visible token there is only one side, and its value is carried
        over: the nearest thing to interpolation an end allows.
        """
        return np.interp(at, self.times, self.values)

    def nearest(self, at: NDArray[np.float64]) -> NDArray[np.float64]:
        """The value of the visible token nearest in time to each instant of ``at``."""
        after = np.searchsorted(self.times, at)
        before = np.clip(after - 1, 0, len(self.times) - 1)
        after = np.clip(after, 0, len(self.times) - 1)
        closer_after = np.abs(self.times[after] - at) < np.abs(self.times[before] - at)
        return self.values[np.where(closer_after, after, before)]

    def __len__(self) -> int:
        return len(self.times)
