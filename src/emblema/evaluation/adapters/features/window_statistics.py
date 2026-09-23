from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class WindowStatistics:
    """What one window says about each channel it holds, as a row of numbers per channel.

    The bridge between a set of observations and the table every classical method needs. A
    window is neither rectangular nor on a grid — the channels present differ and so do the
    instants — so nothing about it can be fed to a tree until it has been summarised, and this
    is where the summarising happens once for every scheme that builds on it.

    A statistic a channel cannot support is ``NaN`` rather than a stand-in: a channel observed
    once has no slope and no step between readings, and filling those with zeros would tell a
    tree that the channel was flat. Gradient boosting learns a direction for a missing value,
    so the absence is information rather than a hole to be patched.

    Attributes:
        channels: Identifiers of the channels the window holds, ascending.
        rows: One row per channel, one column per name in ``NAMES``.
    """

    channels: NDArray[np.int64]
    rows: NDArray[np.float64]

    NAMES: ClassVar[tuple[str, ...]] = (
        "count",
        "mean",
        "deviation",
        "minimum",
        "maximum",
        "first",
        "last",
        "slope",
        "step",
        "gap",
    )

    @classmethod
    def of(cls, window: TokenWindow) -> Self:
        """Summarise ``window`` channel by channel.

        The window's canonical order puts the tokens of one channel in time order already, so
        grouping by channel with a stable sort leaves each group ordered in time and the first
        and last readings are the ends of its run.
        """
        ids = np.asarray(window.channel_ids, dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        channels, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
        values = np.asarray(window.values, dtype=np.float64)[order]
        times = np.asarray(window.times, dtype=np.float64)[order]
        gaps = np.asarray(window.gaps, dtype=np.float64)[order]
        with np.errstate(invalid="ignore", divide="ignore"):
            rows = np.column_stack(
                (
                    counts.astype(np.float64),
                    *cls._of_values(values, starts, counts),
                    cls._slopes(values, times, starts, counts),
                    cls._steps(values, starts, counts),
                    np.add.reduceat(gaps, starts) / counts,
                )
            )
        return cls(channels=channels, rows=rows)

    @staticmethod
    def _of_values(
        values: NDArray[np.float64], starts: NDArray[np.int64], counts: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], ...]:
        """Mean, deviation, extremes and ends of each channel's readings."""
        mean = np.add.reduceat(values, starts) / counts
        variance = np.add.reduceat(values * values, starts) / counts - mean * mean
        return (
            mean,
            np.sqrt(np.maximum(variance, 0.0)),
            np.minimum.reduceat(values, starts),
            np.maximum.reduceat(values, starts),
            values[starts],
            values[starts + counts - 1],
        )

    @staticmethod
    def _slopes(
        values: NDArray[np.float64],
        times: NDArray[np.float64],
        starts: NDArray[np.int64],
        counts: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Least-squares slope of each channel's readings against time within the window.

        A channel observed at one instant, or at one instant repeatedly, has no slope: the
        denominator is the spread of its times and there is none.
        """
        mean_time = np.add.reduceat(times, starts) / counts
        mean_value = np.add.reduceat(values, starts) / counts
        spread = np.add.reduceat(times * times, starts) / counts - mean_time * mean_time
        covariance = np.add.reduceat(times * values, starts) / counts - mean_time * mean_value
        return np.where(spread > 0.0, covariance / spread, np.nan)

    @staticmethod
    def _steps(
        values: NDArray[np.float64], starts: NDArray[np.int64], counts: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """Mean absolute move between one reading of a channel and the next.

        The differences are taken over the whole sorted run and the ones that straddle two
        channels are zeroed, which is cheaper than slicing each channel out and gives the same
        sums. A channel observed once has no move to average.
        """
        within = np.concatenate((np.abs(np.diff(values)), [0.0]))
        within[starts[1:] - 1] = 0.0
        moves = np.add.reduceat(within, starts) / np.maximum(counts - 1, 1)
        return np.where(counts > 1, moves, np.nan)
