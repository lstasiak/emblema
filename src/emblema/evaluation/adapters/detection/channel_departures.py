import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import OverlappingWindowsError
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens

Floats = NDArray[np.float64]
Indices = NDArray[np.int64]

_FEATURES = BinnedTokens.FEATURE_NAMES
_MEAN, _MIN, _MAX, _LAST = (
    _FEATURES.index(name) for name in ("mean", "minimum", "maximum", "last")
)
_PRESENCE, _LOG_COUNT = BinnedTokens.PRESENCE, BinnedTokens.LOG_COUNT


@dataclass(frozen=True)
class ChannelDepartures:
    """How far each channel of a window lies from the same channel over the windows before it.

    A channel's window is summarised by five values: the mean of its spans' means, the least
    minimum, the largest maximum, the last reading and the log of the readings counted. Its
    history is the windows that end one, two and up to ``span - 1`` window lengths earlier,
    each the window ending nearest that instant, within half a length, so a window published a
    few seconds off the others still takes its place. The departure is the distance from the
    history's median over its scaled median absolute deviation, the largest over the five
    values, and over the spans when the spans depart; the spread is floored at a tenth of the
    channel's spread over the fitting side, so a quantised channel whose recent windows happen
    to agree does not read a step of one count as an event. A channel absent from the window,
    or with fewer than half its history holding a reading, departs nought.

    The history is drawn from the windows handed in and nothing else, so it never reaches a
    window its caller did not read. Windows of several units are one history, as the months of
    one mission are, so a month's first windows read the last of the month before; units whose
    windows interleave in time are refused, since their ends lie on separate axes and a history
    over them would mix them.

    Attributes:
        history: How far back a channel is read and what departs.
        summaries: ``[windows, channels, 5]``, missing where the channel holds nothing.
        bins: ``[windows, channels, bins, 5]``, missing where a span holds nothing; ``None``
            unless the spans depart.
        before: ``[windows, span - 1]``, the window ending one, two and more lengths earlier,
            ``-1`` where none does.
    """

    DEVIATION_PER_MAD: ClassVar[float] = 1.4826
    FLOOR_SHARE: ClassVar[float] = 0.1
    VALUES: ClassVar[int] = 5
    # Chunks bound the memory of the history gathered at once, a sixth as many windows when
    # each brings its spans.
    CHUNK: ClassVar[int] = 1024

    history: ChannelHistory
    summaries: Floats
    bins: Floats | None
    before: Indices

    @classmethod
    def laid(
        cls,
        history: ChannelHistory,
        cells: Iterable[Floats],
        *,
        ends_at: Sequence[float],
        units: Sequence[str],
        window_length: float,
    ) -> Self:
        """The departures over windows laid as ``cells``, chunk after chunk in window order.

        Each chunk is ``[windows, channels, bins, features]`` in the layout's feature order;
        ``ends_at`` is when each window ends and ``units`` whose it is, in the same order.

        Raises:
            OverlappingWindowsError: If the windows of two units interleave in time.
            ValueError: If the cells, the ends and the units name different numbers of windows.
        """
        ends = np.asarray(ends_at, dtype=np.float64)
        if len(ends) != len(units):
            raise ValueError(f"{len(ends)} ends were given for {len(units)} units")
        # Refused before a cell is laid, since laying reads the whole block.
        cls.refuse_interleaved(ends, units)
        summaries: list[Floats] = []
        bins: list[Floats] = []
        for chunk in cells:
            summaries.append(cls.summaries_of(chunk))
            if history.departure is DepartureCells.BINS:
                bins.append(cls.bin_values_of(chunk))
        laid = np.concatenate(summaries)
        if len(laid) != len(ends):
            raise ValueError(f"{len(laid)} windows were laid for {len(ends)} ends")
        return cls(
            history=history,
            summaries=laid,
            bins=np.concatenate(bins) if bins else None,
            before=cls.windows_before(
                ends,
                length=window_length,
                count=history.windows_before,
            ),
        )

    @staticmethod
    def summaries_of(cells: Floats) -> Floats:
        """``[windows, channels, 5]``: each channel's window summarised over its spans."""
        present = cells[..., _PRESENCE] > 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            count = present.sum(axis=-1)
            mean = np.where(present, cells[..., _MEAN], 0.0).sum(axis=-1) / count
            least = np.where(present, cells[..., _MIN], np.inf).min(axis=-1)
            most = np.where(present, cells[..., _MAX], -np.inf).max(axis=-1)
            last_bin = cells.shape[-2] - 1 - np.argmax(present[..., ::-1], axis=-1)
            last = np.take_along_axis(cells[..., _LAST], last_bin[..., None], axis=-1)[..., 0]
            readings = np.where(present, np.expm1(cells[..., _LOG_COUNT]), 0.0).sum(axis=-1)
        summaries = np.stack((mean, least, most, last, np.log1p(readings)), axis=-1)
        laid: Floats = np.where((count > 0)[..., None], summaries, np.nan)
        return laid

    @staticmethod
    def bin_values_of(cells: Floats) -> Floats:
        """``[windows, channels, bins, 5]``: each span's five values, missing where it is empty."""
        present = cells[..., _PRESENCE] > 0.0
        values = np.stack(
            (
                cells[..., _MEAN],
                cells[..., _MIN],
                cells[..., _MAX],
                cells[..., _LAST],
                cells[..., _LOG_COUNT],
            ),
            axis=-1,
        )
        laid: Floats = np.where(present[..., None], values, np.nan)
        return laid

    @staticmethod
    def refuse_interleaved(ends: Floats, units: Sequence[str]) -> None:
        """Refuse units whose windows interleave in time, the ends of each on an axis of its own.

        Raises:
            OverlappingWindowsError: If a unit's first window ends at or before another's last.
        """
        keys = np.asarray([str(unit) for unit in units])
        spans = sorted(
            (float(ends[keys == unit].min()), float(ends[keys == unit].max()), str(unit))
            for unit in np.unique(keys)
        )
        latest, holder = -np.inf, ""
        for first, last, unit in spans:
            if first <= latest:
                raise OverlappingWindowsError(
                    f"the windows of {holder} and {unit} interleave in time, so their ends lie "
                    "on axes of their own and neither is the other's history"
                )
            if last > latest:
                latest, holder = last, unit

    @staticmethod
    def windows_before(ends: Floats, *, length: float, count: int) -> Indices:
        """``[windows, count]``: the window ending nearest each of the ``count`` lengths before.

        A place with no window ending within half a length of it is ``-1``; where two do, the
        nearer, the earlier on a tie.
        """
        order = np.argsort(ends, kind="stable")
        ordered = ends[order]
        found = np.full((len(ends), count), -1, dtype=np.int64)
        if not len(ends):
            return found
        for back in range(1, count + 1):
            target = ends - back * length
            after = np.clip(np.searchsorted(ordered, target), 0, len(ordered) - 1)
            earlier = np.clip(after - 1, 0, len(ordered) - 1)
            nearer = np.where(
                np.abs(ordered[earlier] - target) <= np.abs(ordered[after] - target),
                earlier,
                after,
            )
            close = np.abs(ordered[nearer] - target) < length / 2
            found[:, back - 1] = np.where(close, order[nearer], -1)
        return found

    def present(self, indices: Indices) -> NDArray[np.bool_]:
        """``[len(indices), channels]``: whether the channel holds a reading in the window."""
        held: NDArray[np.bool_] = ~np.isnan(self.summaries[indices, :, 0])
        return held

    def floor_over(self, fitting: Indices) -> Floats:
        """``[channels, 5]``: the least spread of a history, from the fitting windows' summaries.

        A channel and value without spread on the fitting side takes the share of the median
        spread over those that have one; where none has, one.
        """
        with np.errstate(invalid="ignore"):
            spread = np.nan_to_num(self._spread(self.summaries[fitting], axis=0), nan=0.0)
        above = spread[spread > 0.0]
        typical = float(np.median(above)) if len(above) else 1.0 / self.FLOOR_SHARE
        floor: Floats = self.FLOOR_SHARE * np.where(spread > 0.0, spread, typical)
        return floor

    def of(self, indices: Indices, floor: Floats) -> Floats:
        """``[len(indices), channels]``: each channel's departure from its history."""
        spans = 1 if self.bins is None else self.bins.shape[2]
        chunk = max(1, self.CHUNK // spans)
        parts = [
            self._departures(indices[start : start + chunk], floor)
            for start in range(0, len(indices), chunk)
        ]
        if not parts:
            return np.zeros((0, self.summaries.shape[1]), dtype=np.float64)
        return np.concatenate(parts)

    def _departures(self, indices: Indices, floor: Floats) -> Floats:
        before = self.before[indices]
        found = before >= 0
        if self.bins is None:
            context = np.where(
                found[:, :, None, None], self.summaries[np.maximum(before, 0)], np.nan
            )
            now = self.summaries[indices][:, :, None, :]
            per_window = 1
        else:
            gathered = np.where(
                found[:, :, None, None, None], self.bins[np.maximum(before, 0)], np.nan
            )
            windows, earlier, channels, spans, values = gathered.shape
            context = gathered.transpose(0, 1, 3, 2, 4).reshape(
                windows, earlier * spans, channels, values
            )
            now = self.bins[indices]
            per_window = spans
        centre = self._median(context, axis=1)
        spread = self._spread(context, axis=1)
        windows, samples, channels, values = context.shape
        held = (~np.isnan(context)).reshape(
            windows, samples // per_window, per_window, channels, values
        )
        loaded = held.any(axis=(2, 4)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.maximum(np.nan_to_num(spread, nan=0.0), floor[None])
            departures = np.abs(now - centre[:, :, None, :]) / scale[:, :, None, :]
            largest = np.nanmax(np.where(np.isnan(departures), -np.inf, departures), axis=(2, 3))
        enough = loaded >= self.history.least_history
        departed: Floats = np.where(enough & np.isfinite(largest), largest, 0.0)
        return departed

    @staticmethod
    def _median(values: Floats, axis: int, *, keepdims: bool = False) -> Floats:
        """The median along ``axis`` over the values present; missing where none is, silently."""
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            median: Floats = np.nanmedian(values, axis=axis, keepdims=keepdims)
        return median

    @classmethod
    def _spread(cls, values: Floats, axis: int) -> Floats:
        """The scaled median absolute deviation along ``axis``, missing values left out."""
        centre = cls._median(values, axis=axis, keepdims=True)
        spread: Floats = cls.DEVIATION_PER_MAD * cls._median(np.abs(values - centre), axis=axis)
        return spread
