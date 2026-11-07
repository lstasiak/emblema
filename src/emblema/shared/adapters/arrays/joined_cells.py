from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.laid_window import LaidWindow

_PRESENCE = BinnedTokens.PRESENCE
_LOG_COUNT = BinnedTokens.LOG_COUNT
_MEAN, _MINIMUM, _MAXIMUM, _FIRST, _LAST, _OFFSET = range(2, 8)


@dataclass(frozen=True)
class JoinedCells:
    """The cells of a window that spans consecutive windows, joined from each window's own cells.

    A window with history is the windows before it and itself, side by side in time. Laying it
    out from their readings concatenated gives the same cells as laying each window once and
    joining their cells, where every bin gathers whole cells of the windows: its count, level,
    extremes, ends and the offset of its last reading follow from theirs, and the carry-forward
    and the "since" count run over the joined bins as they would over one window's. Each window
    is then laid once, however many windows with history it stands in. The equality holds where
    every window is laid over the same channels and as many bins, and a bin of the joined window
    gathers whole cells. The timed cells alone are joined; a window with history holds no static
    feature.

    Attributes:
        bins: The bins of the joined window; the cells joined must cut into them.
    """

    bins: int

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError(f"a joined window has at least one bin, got {self.bins}")

    def join(self, cells: NDArray[np.float64]) -> NDArray[np.float64]:
        """``[..., bins, features]`` from ``[..., cells, features]``, the windows' cells in time.

        A window that is not there, before a unit's start or with no reading, stands as cells of
        nothing: no presence, no count.

        Raises:
            ValueError: If the cells do not cut into the bins.
        """
        *lead, count, features = cells.shape
        if count % self.bins:
            raise ValueError(f"{count} cells do not cut into {self.bins} bins")
        each = count // self.bins
        grouped = cells.reshape(*lead, self.bins, each, features)
        present = grouped[..., _PRESENCE] > 0.0
        observed = present.any(axis=-1)
        counts = np.where(present, np.rint(np.expm1(grouped[..., _LOG_COUNT])), 0.0)
        count_of = counts.sum(axis=-1)
        mean = np.where(
            observed,
            (counts * grouped[..., _MEAN]).sum(axis=-1) / np.where(count_of > 0, count_of, 1.0),
            0.0,
        )
        minimum = np.where(
            observed, np.where(present, grouped[..., _MINIMUM], np.inf).min(axis=-1), 0.0
        )
        maximum = np.where(
            observed, np.where(present, grouped[..., _MAXIMUM], -np.inf).max(axis=-1), 0.0
        )
        earliest = present.argmax(axis=-1)[..., None]
        latest_cell = (each - 1 - present[..., ::-1].argmax(axis=-1))[..., None]
        first = np.take_along_axis(grouped[..., _FIRST], earliest, axis=-1)[..., 0]
        last = np.take_along_axis(grouped[..., _LAST], latest_cell, axis=-1)[..., 0]
        offset = (
            latest_cell[..., 0]
            + np.take_along_axis(grouped[..., _OFFSET], latest_cell, axis=-1)[..., 0]
        ) / each
        offset = np.where(observed, np.minimum(offset, 1.0), 0.0)
        first, last = np.where(observed, first, 0.0), np.where(observed, last, 0.0)
        places = np.arange(self.bins)
        latest = np.maximum.accumulate(np.where(observed, places, -1), axis=-1)
        held = np.where(latest >= 0, np.take_along_axis(last, np.maximum(latest, 0), axis=-1), 0.0)
        since = np.where(latest >= 0, places - latest, places + 1) / self.bins
        for feature in (mean, minimum, maximum, first, last):
            feature[~observed] = held[~observed]
        joined: NDArray[np.float64] = np.stack(
            (
                observed.astype(np.float64),
                np.log1p(count_of),
                mean,
                minimum,
                maximum,
                first,
                last,
                offset,
                since,
            ),
            axis=-1,
        )
        return joined

    @staticmethod
    def laid(
        cells: NDArray[np.float64], rows: NDArray[np.int64], *, own_channels: bool
    ) -> LaidWindow:
        """One joined window's cells ``[rows, bins, features]`` as the layout lays a window out.

        Over the window's own channels, a row no bin observed is dropped, as the layout gives a
        window a row only for a channel it holds; over the corpus's, every row stays.
        """
        if own_channels:
            kept = (cells[..., _PRESENCE] > 0.0).any(axis=-1)
            cells, rows = cells[kept], rows[kept]
        channels, bins, features = cells.shape
        return LaidWindow(
            features=cells.transpose(1, 0, 2).reshape(-1, features),
            channel_ids=np.tile(rows, bins),
            times=np.repeat((np.arange(bins) + 0.5) / bins, channels),
            timeless=np.zeros(channels * bins, dtype=bool),
        )
