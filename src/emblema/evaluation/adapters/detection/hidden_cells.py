from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.detection.masked_cells import MaskedCells
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens

_VALUES = BinnedTokens.VALUE_FEATURES
_FIRST = BinnedTokens.FEATURE_NAMES.index("first")
_LAST = BinnedTokens.FEATURE_NAMES.index("last")
_MEAN = BinnedTokens.FEATURE_NAMES.index("mean")


@dataclass(frozen=True)
class HiddenCells:
    """The timed cells of one draw as a grid of bins by channels, and what they are scored on.

    A window's timed cells stand bin by bin with its channels within a bin, so they fold into a
    ``[bins, channels]`` grid. The cells a draw scores are the hidden ones that held a reading:
    a static cell has no time and is not scored, and a hidden cell that held nothing has no value
    to miss. What a channel's visible bins say about a hidden one is read here once, for every
    answer that learnt nothing: the last visible value carried forward, and the line from the
    last visible value before the cell to the first after it.

    Attributes:
        channels: ``[channels]`` — the channel of every column of the grid.
        visible: ``[bins, channels, features]`` — the visible layout of the timed cells.
        targets: ``[bins, channels, 5]`` — the five values of every cell, from every reading.
        scored: ``[bins, channels]`` — the hidden cells that held a reading.
        carried: ``[bins, channels]`` — the channel's last visible value before the cell.
        interpolated: ``[bins, channels]`` — the line through the visible values on either side,
            the one side's value where only one has any.
        before: ``[bins, channels]`` — whether a visible bin of the channel lies before the cell.
        either: ``[bins, channels]`` — whether one lies on either side or both.
    """

    channels: NDArray[np.int64]
    visible: NDArray[np.float64]
    targets: NDArray[np.float64]
    scored: NDArray[np.bool_]
    carried: NDArray[np.float64]
    interpolated: NDArray[np.float64]
    before: NDArray[np.bool_]
    either: NDArray[np.bool_]

    @classmethod
    def of(cls, cells: MaskedCells, bins: int) -> Self:
        """The draw's timed cells on ``bins`` spans.

        Raises:
            ValueError: If the timed cells do not fill ``bins`` spans of the same channels.
        """
        timed = ~cells.timeless
        count = int(timed.sum())
        if count == 0 or count % bins:
            raise ValueError(f"{count} timed cells do not lie on {bins} bins of the same channels")
        width = count // bins

        def grid(values: NDArray[np.generic]) -> NDArray[np.generic]:
            return values[timed].reshape(bins, width, *values.shape[1:])

        full = grid(cells.full).astype(np.float64)
        visible = grid(cells.visible).astype(np.float64)
        hidden = grid(cells.hidden).astype(np.bool_)
        held = visible[..., BinnedTokens.PRESENCE] > 0.5
        steps = np.arange(bins)[:, None]
        last_before = np.maximum.accumulate(np.where(held, steps, -1), axis=0)
        # The last visible bin at or before each cell, then shifted so a cell does not count
        # itself; a hidden cell is never held, so only the visible bins before it remain.
        previous = np.vstack((np.full((1, width), -1), last_before[:-1]))
        first_after = np.minimum.accumulate(np.where(held, steps, bins)[::-1], axis=0)[::-1]
        following = np.vstack((first_after[1:], np.full((1, width), bins)))
        before = previous >= 0
        after = following < bins
        columns = np.arange(width)[None, :]
        last = np.where(before, visible[np.maximum(previous, 0), columns, _LAST], 0.0)
        first = np.where(after, visible[np.minimum(following, bins - 1), columns, _FIRST], 0.0)
        span = np.where(before & after, following - previous, 1)
        line = last + (first - last) * (steps - previous) / span
        interpolated = np.where(before & after, line, np.where(before, last, first))
        return cls(
            channels=cells.channel_ids[timed][:width].astype(np.int64),
            visible=visible,
            targets=full[..., _VALUES],
            scored=hidden & (full[..., BinnedTokens.PRESENCE] > 0.5),
            carried=last,
            interpolated=np.where(before | after, interpolated, 0.0),
            before=before,
            either=before | after,
        )

    @property
    def visible_means(self) -> NDArray[np.float64]:
        """``[bins, channels]`` — every visible cell's mean, zero where nothing visible was read."""
        held = self.visible[..., BinnedTokens.PRESENCE] > 0.5
        return np.where(held, self.visible[..., _MEAN], 0.0)

    @property
    def visible_presence(self) -> NDArray[np.bool_]:
        """``[bins, channels]`` — the cells left visible that held a reading."""
        return self.visible[..., BinnedTokens.PRESENCE] > 0.5

    def interpolation(self) -> NDArray[np.float64]:
        """``[bins, channels, 5]`` — the interpolated value as every one of the five values."""
        return np.repeat(self.interpolated[..., None], 5, axis=-1)
