from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from torch import Tensor

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class GridReading:
    """How a network reads windows off the grid: over which steps, and which channels.

    The channels are the ones the windows it was trained on observed, picked by the grid's own
    rule, so a model is not made as wide as a vocabulary most of which its task never reports.
    Every channel read brings its mask along, whether or not the training windows ever missed a
    step of it: a network's input has a place for a mask per channel, and a later window may
    miss a step the training windows did not.

    Attributes:
        steps: Equal steps a window is laid on.
        channels: Channels of the corpus's vocabulary a window is first laid over.
        held: The channels read, as rows of that grid, in vocabulary order.
    """

    steps: int
    channels: int
    held: tuple[int, ...]

    @classmethod
    def over(cls, windows: Sequence[TokenWindow], *, steps: int, channels: int) -> Self:
        """The reading the ``windows`` give something to read.

        Never empty: a window holds at least one timed reading, so some channel is observed.

        Raises:
            UnreadableTaskCorpusError: If a window holds a channel the corpus does not name.
        """
        grid = RegularGrid(steps, channels)
        rows = grid.rows_read(grid.of(windows))
        return cls(steps=steps, channels=channels, held=tuple(int(r) for r in rows if r < channels))

    def tensors(self, windows: Sequence[TokenWindow]) -> tuple[Tensor, Tensor]:
        """The values and the masks of every window, ``[windows, held, steps]`` each, on the host.

        Raises:
            UnreadableTaskCorpusError: If a window holds a channel the corpus does not name.
        """
        laid = RegularGrid(self.steps, self.channels).of(windows)
        held = np.asarray(self.held, dtype=np.int64)
        values = torch.from_numpy(laid[:, held].astype(np.float32))
        observed = torch.from_numpy(laid[:, self.channels + held].astype(np.float32))
        return values, observed
