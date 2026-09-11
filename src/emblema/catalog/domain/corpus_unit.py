"""A unit of a corpus and the span of its time axis."""

import math
from collections import Counter
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusUnitError, InvalidTimeExtentError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.static_feature import StaticFeature


@dataclass(frozen=True)
class TimeExtent:
    """A half-open span ``[start, end)`` on a unit's time axis.

    The extent of a unit is a fact the reader states, not the span of its observations: a hospital
    stay covers its protocol's 48 hours whether or not the last hour holds a measurement, and an
    engine of ``L`` cycles spans ``[1, L + 1)`` because a cycle occupies the unit interval that
    starts at its index. Windows are laid over the extent, so it decides how many there are.

    Attributes:
        start: First instant inside the span; finite.
        end: First instant beyond the span; finite, after ``start``.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise InvalidTimeExtentError("start and end must be finite")
        if self.start >= self.end:
            raise InvalidTimeExtentError(f"start must precede end, got [{self.start}, {self.end})")

    @property
    def length(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        return self.start <= time < self.end


@dataclass(frozen=True)
class CorpusUnit:
    """An independent unit of a corpus: an engine, a machine, a patient stay.

    Units are what a corpus is split on, so nothing learned from one unit's data may reach a
    window of another. The unit carries its identity, its time extent and its static features;
    its observations are streamed separately by the reader, because a unit may hold more of them
    than fit in memory.

    Attributes:
        key: Identity of the unit within its corpus.
        extent: Span of the unit's time axis that windows are laid over.
        static_features: Values describing the whole unit, one per channel.
    """

    key: UnitKey
    extent: TimeExtent
    static_features: tuple[StaticFeature, ...] = ()

    def __post_init__(self) -> None:
        counts = Counter(feature.channel for feature in self.static_features)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise InvalidCorpusUnitError(f"duplicate static feature channels: {duplicates}")
