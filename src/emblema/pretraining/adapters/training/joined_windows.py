from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

from emblema.pretraining.domain.exceptions import InvalidTrainingCorpusError
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.window_place import WindowPlace
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
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
    """The windows of one side of a corpus, each joined with the ``history - 1`` before it.

    Every window of both sides is read once and laid into cells, over the channels the corpus
    observes, and a window's history is gathered from there: the windows ending one, two and
    more lengths before it on its axis, whichever side holds them, as an hour of the validation
    side reads the training hours before it. A place with no window ending within a hundredth of
    a length of it is a window of nothing, as a downlink's gap or the start of a stream leaves.
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
    def sides_of(
        cls, corpus: TrainingCorpus, *, history: int, bins: int
    ) -> tuple["JoinedWindows", "JoinedWindows"]:
        """The training and validation windows of ``corpus``, each joined with its history.

        Raises:
            InvalidTrainingCorpusError: If the corpus does not say where its windows lie, its
                windows are not all one length, or two windows of one axis overlap.
        """
        if not corpus.placed:
            raise InvalidTrainingCorpusError(
                f"{corpus.name!r} does not say where its windows lie, so none can be joined "
                "with those before it"
            )
        if bins % history:
            raise InvalidTrainingCorpusError(f"{history} windows do not cut into {bins} bins")
        places = [*corpus.training_places, *corpus.validation_places]
        laid, rows = cls._laid_once((corpus.training, corpus.validation), bins // history)
        positions = cls._history_of(places, history)
        split = len(corpus.training)
        return cls(laid, rows, positions[:split]), cls(laid, rows, positions[split:])

    def __len__(self) -> int:
        return len(self._history)

    @overload
    def __getitem__(self, index: int) -> JoinedWindow: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[JoinedWindow]: ...

    def __getitem__(self, index: int | slice) -> JoinedWindow | Sequence[JoinedWindow]:
        if isinstance(index, slice):
            return JoinedWindows(self._laid, self._rows, self._history[index])
        positions = self._history[index]
        gathered = np.where(
            (positions >= 0)[:, None, None, None], self._laid[np.maximum(positions, 0)], 0.0
        )
        windows, rows, bins, features = gathered.shape
        cells = gathered.transpose(1, 0, 2, 3).reshape(rows, windows * bins, features)
        return JoinedWindow(cells=cells, rows=self._rows)

    @staticmethod
    def _laid_once(
        sides: Sequence[Sequence[TokenWindow]], bins: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """``[windows, rows, bins, features]``: every window of the sides, in order, laid once.

        The windows are read a slice at a time, so no more of them stand as objects at once than
        a slice holds: a corpus of tens of thousands of hours would not fit otherwise. Each is
        laid over its own channels, then placed among the rows of
        every channel the windows observe, where a channel it does not hold is cells of nothing,
        as a layout over the corpus's channels would leave it.

        Raises:
            InvalidTrainingCorpusError: If a window holds a static feature, which no window
                joined with others carries.
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
                    raise InvalidTrainingCorpusError(
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
    def _history_of(places: Sequence[WindowPlace], history: int) -> NDArray[np.int64]:
        """``[windows, history]``: each window's history by position, earliest first.

        Raises:
            InvalidTrainingCorpusError: If the windows are not all one length, or two windows of
                one axis overlap.
        """
        lengths = {round(place.end - place.start, 9) for place in places}
        if len(lengths) != 1:
            raise InvalidTrainingCorpusError(
                f"windows of {len(lengths)} lengths cannot be joined with those before them"
            )
        (length,) = lengths
        by_axis: dict[str, list[int]] = defaultdict(list)
        for position, place in enumerate(places):
            by_axis[place.axis].append(position)
        out = np.full((len(places), history), -1, dtype=np.int64)
        for axis, members in by_axis.items():
            ends = np.asarray([places[m].end for m in members])
            order = np.argsort(ends, kind="stable")
            sorted_ends, sorted_members = ends[order], np.asarray(members)[order]
            if np.any(np.diff(sorted_ends) < length * (1 - _TOLERANCE)):
                raise InvalidTrainingCorpusError(
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
