from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChannelErrors:
    """What a detector scored by an error missed in every window, channel by channel.

    One row per window and one column per channel of the grid: the loss summed over the
    channel's scored cells across every draw, and how many cells that was. The two readings a
    recipe can ask for come off these sums: the raw error, one mean over all the window's
    cells, and the error per channel, a mean per channel that a scale per channel then divides.

    Invariants: the sums and the counts have one row per window and one column per channel.

    Attributes:
        channels: ``[channels]`` — the channel of every column.
        missed: ``[windows, channels]`` — the loss summed over the scored cells.
        counted: ``[windows, channels]`` — how many cells were scored.
    """

    channels: NDArray[np.int64]
    missed: NDArray[np.float64]
    counted: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.missed.shape != self.counted.shape or self.missed.shape[1:] != self.channels.shape:
            raise ValueError(
                f"the sums {self.missed.shape}, the counts {self.counted.shape} and the channels "
                f"{self.channels.shape} do not lie on one grid"
            )

    @property
    def unread(self) -> NDArray[np.bool_]:
        """``[windows]`` — the windows that held no scored cell in any draw."""
        unread: NDArray[np.bool_] = self.counted.sum(axis=1) == 0
        return unread

    def raw(self) -> NDArray[np.float64]:
        """``[windows]`` — the mean loss over every scored cell; zero where none was."""
        missed = self.missed.sum(axis=1)
        counted = self.counted.sum(axis=1)
        return np.divide(missed, counted, out=np.zeros_like(missed), where=counted > 0)

    def per_channel(self) -> NDArray[np.float64]:
        """``[windows, channels]`` — the mean loss per channel; not a number where none scored."""
        with np.errstate(invalid="ignore", divide="ignore"):
            means: NDArray[np.float64] = self.missed / self.counted
        return means
