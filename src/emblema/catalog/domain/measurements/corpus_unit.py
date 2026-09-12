from collections import Counter
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent


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
