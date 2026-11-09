from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.exceptions import (
    OverlappingJoinedWindowsError,
    UnjoinableWindowsError,
)
from emblema.shared.kernel.tokens import TokenWindow

# Windows are laid this many at a time while every published window is laid once.
_LAID_AT_ONCE = 4096
# How far from its place in a history a window may end and still fill it, as a share of a window:
# a month published seconds off the hour stays on the grid (36 s of an hour), while a window
# further off would shift its readings within the joined bins, so it is left out instead.
_TOLERANCE = 0.01


@dataclass(frozen=True)
class JoinedWindow:
    """One window joined with those before it on its axis: their cells, side by side in time.

    Attributes:
        cells: ``[rows, history * bins_per_window, features]``, the published windows' cells in
            time order, the window's own last; a window that is not there is cells of nothing.
        rows: ``[rows]``, the channel identifier of each row.
    """

    cells: NDArray[np.float64]
    rows: NDArray[np.int64]


class JoinedWindows(Sequence[JoinedWindow]):
    """Published windows, each joined with the ``history - 1`` before it on its axis.

    Every window is read once and laid into cells, over the channels the windows observe, and a
    window's history is gathered from there: the windows ending one, two and more lengths
    before it on its axis, wherever they are among the windows given, as a pretraining's
    validation hour reads the training hours before it and a scored hour the fitting ones. A
    place with no window ending within a hundredth of a length of it is a window of nothing, as
    a downlink's gap or the start of a stream leaves. Pretraining and evaluation both join here,
    so a backbone is evaluated over the windows it was trained over (ADR-0076).
    """

    def __init__(
        self, laid: NDArray[np.float64], rows: NDArray[np.int64], history: NDArray[np.int64]
    ) -> None:
        """Windows over ``laid`` ``[windows, rows, bins, features]`` by ``history``'s positions.

        ``history`` is ``[windows, history]``: each window's positions in ``laid``, earliest
        first and its own last, ``-1`` where none.
        """
        self._laid = laid
        self._rows = rows
        self._history = history

    @classmethod
    def of(
        cls,
        sides: Sequence[Sequence[TokenWindow]],
        *,
        axes: Sequence[str],
        starts: Sequence[float],
        ends: Sequence[float],
        history: int,
        bins: int,
    ) -> Self:
        """The windows of ``sides`` in order, each joined with its history over ``bins`` bins.

        ``axes``, ``starts`` and ``ends`` say where each window lies, in the same order.

        Raises:
            OverlappingJoinedWindowsError: If two windows of one axis overlap.
            UnjoinableWindowsError: If no windows are given, the places are not one per window,
                the history does not cut into the bins, the windows are not all one length, or a
                window holds a static feature.
        """
        count = sum(len(side) for side in sides)
        if not count:
            raise UnjoinableWindowsError("no windows were given to join")
        if not len(axes) == len(starts) == len(ends) == count:
            raise UnjoinableWindowsError(f"{len(axes)} places were given for {count} windows")
        if history < 1 or bins % history:
            raise UnjoinableWindowsError(f"{history} windows do not cut into {bins} bins")
        laid, rows = cls._laid_once(sides, bins // history)
        positions = cls._history_of(
            axes, np.asarray(starts, dtype=np.float64), np.asarray(ends, dtype=np.float64), history
        )
        return cls(laid, rows, positions)

    def __len__(self) -> int:
        return len(self._history)

    @overload
    def __getitem__(self, index: int) -> JoinedWindow: ...

    @overload
    def __getitem__(self, index: slice) -> "JoinedWindows": ...

    def __getitem__(self, index: int | slice) -> "JoinedWindow | JoinedWindows":
        if isinstance(index, slice):
            return JoinedWindows(self._laid, self._rows, self._history[index])
        positions = self._history[index]
        gathered = np.where(
            (positions >= 0)[:, None, None, None], self._laid[np.maximum(positions, 0)], 0.0
        )
        windows, rows, bins, features = gathered.shape
        cells = gathered.transpose(1, 0, 2, 3).reshape(rows, windows * bins, features)
        return JoinedWindow(cells=cells, rows=self._rows)

    def at(self, positions: Sequence[int]) -> "JoinedWindows":
        """The windows at those positions, in the order given, sharing the cells laid once."""
        chosen = np.asarray(positions, dtype=np.intp)
        return JoinedWindows(self._laid, self._rows, self._history[chosen])

    @staticmethod
    def _laid_once(
        sides: Sequence[Sequence[TokenWindow]], bins: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """``[windows, rows, bins, features]``: every window of the sides, in order, laid once.

        The windows are read a slice at a time, so no more of them stand as objects at once than
        a slice holds: a corpus of tens of thousands of hours would not fit otherwise. Each is
        laid over its own channels, then placed among the rows of every channel the windows
        observe, where a channel it does not hold is cells of nothing, as a layout over the
        corpus's channels would leave it.

        Raises:
            UnjoinableWindowsError: If a window holds a static feature, which no joined window
                carries.
        """
        layout = BinnedTokens(bins)
        own: list[tuple[NDArray[np.int64], NDArray[np.float64]]] = []
        slices = (
            side[start : start + _LAID_AT_ONCE]
            for side in sides
            for start in range(0, len(side), _LAID_AT_ONCE)
        )
        for chunk in slices:
            for window in layout.lay(chunk):
                if window.timeless.any():
                    raise UnjoinableWindowsError(
                        "a window holding static features cannot be joined with others"
                    )
                count = len(window) // bins
                own.append(
                    (
                        window.channel_ids[:count],
                        window.features.reshape(bins, count, -1).transpose(1, 0, 2),
                    )
                )
        rows = np.unique(np.concatenate([channels for channels, _ in own]))
        laid = np.zeros((len(own), len(rows), bins, BinnedTokens.N_FEATURES))
        for position, (channels, cells) in enumerate(own):
            laid[position, np.searchsorted(rows, channels)] = cells
        return laid, rows.astype(np.int64)

    @staticmethod
    def _history_of(
        axes: Sequence[str],
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        history: int,
    ) -> NDArray[np.int64]:
        """``[windows, history]``: each window's history by position, earliest first.

        Raises:
            UnjoinableWindowsError: If the windows are not all one length.
            OverlappingJoinedWindowsError: If two windows of one axis overlap.
        """
        lengths = np.unique(np.round(ends - starts, 9))
        if len(lengths) != 1:
            raise UnjoinableWindowsError(
                f"windows of {len(lengths)} lengths cannot be joined with each other"
            )
        length = float(lengths[0])
        by_axis: dict[str, list[int]] = defaultdict(list)
        for position, axis in enumerate(axes):
            by_axis[axis].append(position)
        out = np.full((len(axes), history), -1, dtype=np.int64)
        for axis, members in by_axis.items():
            order = np.argsort(ends[members], kind="stable")
            sorted_ends, sorted_members = ends[members][order], np.asarray(members)[order]
            if np.any(np.diff(sorted_ends) < length * (1 - _TOLERANCE)):
                raise OverlappingJoinedWindowsError(
                    f"windows of {axis!r} overlap, so none of them is another's history"
                )
            for back in range(history):
                wanted = sorted_ends - back * length
                found = np.clip(np.searchsorted(sorted_ends, wanted), 0, len(sorted_ends) - 1)
                below = np.clip(found - 1, 0, len(sorted_ends) - 1)
                nearest = np.where(
                    np.abs(sorted_ends[below] - wanted) < np.abs(sorted_ends[found] - wanted),
                    below,
                    found,
                )
                near = np.abs(sorted_ends[nearest] - wanted) <= length * _TOLERANCE
                out[sorted_members, history - 1 - back] = np.where(
                    near, sorted_members[nearest], -1
                )
        return out
