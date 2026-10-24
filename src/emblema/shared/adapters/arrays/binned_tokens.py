from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.tokens import N_CELL_FEATURES, PADDING_CHANNEL_ID, TokenWindow


class BinnedTokens:
    """Lays a window out as one token per channel and span of time, with what the span held.

    The window is cut into ``bins`` equal spans. A channel's row of bins becomes a token per
    bin, timed at the bin's centre, whose features say what the bin held of that channel:
    whether anything was measured, how much, the level, the extremes, the ends, where in the bin
    the last reading fell, and how long the channel has gone unobserved. An empty bin carries the
    channel's last value forward in its value features, zero where it has none, so the encoder
    reads a level beside the mark that nothing was measured; its "since" counts bins from the
    channel's last observed bin, or from the window's start where there is none, a lower bound
    as the raw gap of a first token is. A static feature is one timeless cell, present or not.

    Which channels get a row is the layout's choice. Given none, every channel the window holds;
    given the channels a corpus observes, each of them, so a channel never measured in the
    window stands as a row of empty cells, and a window holding a channel outside them is
    refused. The cells keep the canonical order of tokens, timeless first and then bin by bin
    with the channels within a bin, so the batch is a ``TokenBatch`` whose feature axis is
    ``N_FEATURES`` wide rather than two.

    A window may also be laid out ``without`` some of its cells: the readings that fell in them
    are left out and the rest laid on the same rows and bins, so the cells around a hidden one
    carry forward and count from what remains visible, never from what was hidden.

    Attributes:
        bins: Equal spans the window is cut into.
    """

    N_FEATURES: ClassVar[int] = N_CELL_FEATURES
    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "presence",
        "log_count",
        "mean",
        "minimum",
        "maximum",
        "first",
        "last",
        "last_offset",
        "since",
    )
    # Where on the feature axis the presence, the count and the five values stand: whoever
    # scores or answers a cell reads these, never a literal.
    PRESENCE: ClassVar[int] = FEATURE_NAMES.index("presence")
    LOG_COUNT: ClassVar[int] = FEATURE_NAMES.index("log_count")
    VALUE_FEATURES: ClassVar[slice] = slice(
        FEATURE_NAMES.index("mean"), FEATURE_NAMES.index("last") + 1
    )

    def __init__(
        self,
        bins: int,
        *,
        channels: Sequence[int] | None = None,
        statics: Sequence[int] = (),
    ) -> None:
        """Lay windows on ``bins`` spans, over the window's channels or the corpus's.

        Args:
            bins: How many equal spans a window is cut into.
            channels: The measured channels every window gets a row for; ``None`` for the
                channels each window holds.
            statics: The static features every window gets a cell for, read only with
                ``channels``.

        Raises:
            ValueError: If there is no bin, a channel is named twice or is not positive, or
                static features are named without channels.
        """
        if bins < 1:
            raise ValueError(f"a layout has bins, got {bins}")
        if channels is None and statics:
            raise ValueError("static features are named with the corpus's channels, not alone")
        named = (*(channels or ()), *statics)
        if any(channel <= PADDING_CHANNEL_ID for channel in named):
            raise ValueError("a channel identifier is positive")
        if len(set(named)) != len(named):
            raise ValueError("a channel is named once, as measured or as static")
        self.bins = bins
        self._channels = None if channels is None else np.asarray(sorted(channels), dtype=np.int64)
        self._statics = np.asarray(sorted(statics), dtype=np.int64)

    @property
    def features(self) -> int:
        return self.N_FEATURES

    @property
    def value_features(self) -> slice:
        """The mean, minimum, maximum, first and last of a cell: what a clip bounds."""
        return self.VALUE_FEATURES

    @property
    def over_the_corpus(self) -> bool:
        """Whether every window gets the corpus's channels rather than its own."""
        return self._channels is not None

    @property
    def channels(self) -> tuple[int, ...]:
        """The measured channels every window gets a row for; empty over the window's own."""
        return () if self._channels is None else tuple(int(c) for c in self._channels)

    @property
    def statics(self) -> tuple[int, ...]:
        """The static features every window gets a cell for; empty over the window's own."""
        return tuple(int(c) for c in self._statics)

    def lay(self, windows: Sequence[TokenWindow]) -> list[LaidWindow]:
        """Every window as its cells, in the order given.

        Raises:
            ValueError: If a window holds a channel outside the corpus's.
        """
        return [self._cells(window) for window in windows]

    def without(self, window: TokenWindow, hidden: NDArray[np.bool_]) -> LaidWindow:
        """``window`` laid out as if the readings of its ``hidden`` cells had never been taken.

        ``hidden`` marks cells in the order ``lay`` gives them for this window. The rows and the
        static cells are the window's own, hidden or not, so the result stands cell for cell
        beside the full layout; a hidden cell is then an empty one whose value features carry
        the last reading still visible before it, and the cells after it count their "since"
        from that reading too.

        Raises:
            ValueError: If ``hidden`` does not mark exactly this window's cells, or a window
                holds a channel outside the corpus's.
        """
        columns = _Columns.of(window)
        static_rows, rows = self._rows_of(columns)
        expected = len(static_rows) + len(rows) * self.bins
        if hidden.shape != (expected,):
            raise ValueError(
                f"the hidden cells are marked over {hidden.shape}, the window has {expected} cells"
            )
        static_at = np.searchsorted(static_rows, columns.static_ids)
        timed_at = (
            len(static_rows)
            + self._bin_of(columns.times) * len(rows)
            + np.searchsorted(rows, columns.timed_ids)
        )
        return self._laid(columns.kept(~hidden[static_at], ~hidden[timed_at]), static_rows, rows)

    def batch(self, laid: Sequence[LaidWindow], *, dtype: DTypeLike = np.float32) -> TokenBatch:
        return LaidWindow.padded(laid, features=self.N_FEATURES, dtype=dtype)

    def _cells(self, window: TokenWindow) -> LaidWindow:
        columns = _Columns.of(window)
        static_rows, rows = self._rows_of(columns)
        return self._laid(columns, static_rows, rows)

    def _rows_of(self, columns: "_Columns") -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        rows = self._rows(columns.timed_ids, self._channels, "channel")
        static_rows = self._rows(
            columns.static_ids, self._statics if self.over_the_corpus else None, "static feature"
        )
        return static_rows, rows

    def _laid(
        self, columns: "_Columns", static_rows: NDArray[np.int64], rows: NDArray[np.int64]
    ) -> LaidWindow:
        static_cells = self._static_cells(
            columns.static_ids, columns.static_values, rows=static_rows
        )
        timed_cells = self._timed_cells(
            columns.timed_ids, columns.timed_values, columns.times, rows=rows
        )
        return LaidWindow(
            features=np.concatenate((static_cells, timed_cells)),
            channel_ids=np.concatenate((static_rows, np.tile(rows, self.bins))),
            times=np.concatenate(
                (
                    np.zeros(len(static_rows)),
                    np.repeat((np.arange(self.bins) + 0.5) / self.bins, len(rows)),
                )
            ),
            timeless=np.concatenate(
                (
                    np.ones(len(static_rows), dtype=bool),
                    np.zeros(len(rows) * self.bins, dtype=bool),
                )
            ),
        )

    def _bin_of(self, times: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.minimum(np.floor(times * self.bins).astype(np.int64), self.bins - 1)

    def _static_cells(
        self, ids: NDArray[np.int64], values: NDArray[np.float64], *, rows: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """One cell per static feature summarising what the window holds of it, present or not."""
        features = np.zeros((len(rows), self.N_FEATURES))
        # Timeless tokens stand in channel order, so a stable sort groups each feature's values
        # as they are, and a feature named more than once is summarised as a bin would be.
        held, starts, counts = np.unique(ids, return_index=True, return_counts=True)
        at = np.searchsorted(rows, held)
        features[at, 0] = 1.0
        features[at, 1] = np.log1p(counts)
        features[at, 2] = np.add.reduceat(values, starts) / counts if len(values) else 0.0
        features[at, 3] = np.minimum.reduceat(values, starts) if len(values) else 0.0
        features[at, 4] = np.maximum.reduceat(values, starts) if len(values) else 0.0
        features[at, 5] = values[starts]
        features[at, 6] = values[starts + counts - 1]
        return features

    def _timed_cells(
        self,
        ids: NDArray[np.int64],
        values: NDArray[np.float64],
        times: NDArray[np.float64],
        *,
        rows: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """A row of bins per measured channel, bin by bin and channel by channel within a bin."""
        bins = self.bins
        cell = np.searchsorted(rows, ids) * bins + self._bin_of(times)
        # The window's tokens are in time order, so a stable sort by cell leaves every cell's
        # readings in time order: the first and last of a group are the bin's ends.
        order = np.argsort(cell, kind="stable")
        cell, values, times = cell[order], values[order], times[order]
        filled, starts, counts = np.unique(cell, return_index=True, return_counts=True)
        ends = starts + counts - 1
        shape = (len(rows), bins)
        count = np.zeros(shape)
        count.flat[filled] = counts
        observed = count > 0
        with np.errstate(invalid="ignore"):
            mean, minimum, maximum, first, last, offset = (np.zeros(shape) for _ in range(6))
            if len(values):
                mean.flat[filled] = np.add.reduceat(values, starts) / counts
                minimum.flat[filled] = np.minimum.reduceat(values, starts)
                maximum.flat[filled] = np.maximum.reduceat(values, starts)
                first.flat[filled] = values[starts]
                last.flat[filled] = values[ends]
                offset.flat[filled] = times[ends] * bins - filled % bins
        # The bin a reading at the window's very end falls in is the last one, so its offset
        # may reach one; the bins before it never do.
        offset = np.minimum(offset, 1.0)
        latest = np.maximum.accumulate(np.where(observed, np.arange(bins), -1), axis=1)
        held = np.where(latest >= 0, np.take_along_axis(last, np.maximum(latest, 0), axis=1), 0.0)
        since = np.where(latest >= 0, np.arange(bins) - latest, np.arange(bins) + 1) / bins
        for feature in (mean, minimum, maximum, first, last):
            feature[~observed] = held[~observed]
        features = np.stack(
            (
                observed.astype(np.float64),
                np.log1p(count),
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
        # Bin by bin, and channel by channel within a bin: the canonical order of timed tokens.
        laid: NDArray[np.float64] = features.transpose(1, 0, 2).reshape(-1, self.N_FEATURES)
        return laid

    @staticmethod
    def _rows(
        ids: NDArray[np.int64], named: NDArray[np.int64] | None, what: str
    ) -> NDArray[np.int64]:
        held = np.unique(ids)
        if named is None:
            return held
        outside = np.setdiff1d(held, named)
        if outside.size:
            raise ValueError(
                f"a window holds the {what} {int(outside[0])}, which the corpus does not name"
            )
        return named


@dataclass(frozen=True)
class _Columns:
    """A window's columns split into its static and its timed readings, for the layout to read."""

    static_ids: NDArray[np.int64]
    static_values: NDArray[np.float64]
    timed_ids: NDArray[np.int64]
    timed_values: NDArray[np.float64]
    times: NDArray[np.float64]

    @classmethod
    def of(cls, window: TokenWindow) -> "_Columns":
        ids = np.asarray(window.channel_ids, dtype=np.int64)
        values = np.asarray(window.values, dtype=np.float64)
        times = np.asarray(window.times, dtype=np.float64)
        static = np.asarray(window.timeless, dtype=bool)
        return cls(ids[static], values[static], ids[~static], values[~static], times[~static])

    def kept(self, statics: NDArray[np.bool_], timed: NDArray[np.bool_]) -> "_Columns":
        return _Columns(
            self.static_ids[statics],
            self.static_values[statics],
            self.timed_ids[timed],
            self.timed_values[timed],
            self.times[timed],
        )
