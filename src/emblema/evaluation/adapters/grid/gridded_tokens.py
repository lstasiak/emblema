from collections.abc import Sequence

import numpy as np

from emblema.evaluation.adapters.grid.steps import steps_of
from emblema.shared.kernel.tokens import TokenWindow


class GriddedTokens:
    """Lays a window's readings on equal steps and hands them back as tokens.

    What a method on a grid sees, given to a network that reads tokens, so the same network can
    be fed the raw readings or the gridded ones and nothing else differs. A reading falls in the
    step that holds its instant, by the rule every grid here uses, and where several fall in one
    step the latest stands. From a channel's first reading on, every step holds a token at the
    step's start carrying the channel's latest reading forward; before it, the channel has no
    token, as it has none among the raw readings. Static features are left as they are.

    The gap of a gridded token is the time since the channel's latest real reading, in whole
    steps: the previous token of a gridded channel is always one step back, so a gap measured to
    it would say nothing, while this one is zero exactly where a reading was made and grows while
    the value is carried. It is the token's counterpart of the mask beside a grid's values.
    """

    def __init__(self, steps: int) -> None:
        """Lay windows on ``steps`` equal steps.

        Raises:
            ValueError: If there is no step.
        """
        if steps < 1:
            raise ValueError(f"a grid has steps, got {steps}")
        self._steps = steps

    def of(self, windows: Sequence[TokenWindow]) -> list[TokenWindow]:
        """Every window laid on the grid, in the order given."""
        return [self._laid(window) for window in windows]

    def _laid(self, window: TokenWindow) -> TokenWindow:
        ids = np.asarray(window.channel_ids, dtype=np.int64)
        values = np.asarray(window.values, dtype=np.float64)
        timeless = np.asarray(window.timeless, dtype=bool)
        timed = ~timeless
        channels, rows = np.unique(ids[timed], return_inverse=True)
        cells = rows * self._steps + steps_of(
            np.asarray(window.times, dtype=np.float64)[timed], self._steps
        )
        # The latest reading of a step stands: the window is in time order, so the last
        # occurrence of every cell is its latest reading.
        last = len(cells) - 1 - np.unique(cells[::-1], return_index=True)[1]
        read = np.zeros((len(channels), self._steps), dtype=np.float64)
        observed = np.zeros((len(channels), self._steps), dtype=bool)
        read.flat[cells[last]] = values[timed][last]
        observed.flat[cells[last]] = True
        latest = np.maximum.accumulate(np.where(observed, np.arange(self._steps), -1), axis=1)
        # Step by step, and channel by channel within a step: the canonical order of timed
        # tokens, since channels come out of ``unique`` sorted and no two tokens share both.
        step, row = np.nonzero(latest.T >= 0)
        since = latest[row, step]
        # Static features lead a window in canonical order, so they are its first tokens.
        statics = int(timeless.sum())
        return TokenWindow(
            channel_ids=(*window.channel_ids[:statics], *channels[row].tolist()),
            values=(*window.values[:statics], *read[row, since].tolist()),
            times=(*window.times[:statics], *(step / self._steps).tolist()),
            gaps=(*window.gaps[:statics], *((step - since) / self._steps).tolist()),
            timeless=(*window.timeless[:statics], *([False] * len(step))),
        )
