from collections.abc import Iterator
from typing import Protocol

from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.registry.corpus_description import CorpusDescription


class CorpusReader(Protocol):
    """Reads one corpus from its source: describes it, lists its units, streams their observations.

    An adapter is bound to the location and format of its corpus, the way an artifact store is
    bound to its bucket; the port carries no location, so the hexagon knows nothing about files.
    ``describe`` validates the data, checksums it and counts its units and observations in one
    pass and keeps none of it. A description is a function of the bytes: the same data described
    twice gives equal descriptions, and changed data gives a different checksum. ``read_units``
    and ``read_observations`` deliver the same data piecewise, a unit at a time, so a corpus of
    any size can be tokenised without holding it in memory.
    """

    def describe(self) -> CorpusDescription:
        """Validate the corpus and describe it.

        Raises:
            CorpusDataNotFoundError: If the source holds no data where the adapter expects it.
            MalformedCorpusDataError: If the data is present but violates its own format.
        """
        ...

    def read_units(self) -> Iterator[CorpusUnit]:
        """Every unit of the corpus, keys unique, as many as the description counts.

        Raises:
            CorpusDataNotFoundError: If the source holds no data where the adapter expects it.
            MalformedCorpusDataError: If the data is present but violates its own format.
        """
        ...

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """The observations of one unit in non-decreasing time order, all inside its extent.

        Every channel named is one of the description's timed channels. Summed over all units,
        the observations are as many as the description counts. What the reader can settle
        without reading — a key it could never resolve, a source it cannot reach — it settles
        when the stream is asked for; what only the data can settle surfaces as the stream is
        consumed, so a caller that never iterates may never learn of it.

        Raises:
            UnknownUnitError: If the corpus has no unit with that key.
            CorpusDataNotFoundError: If the source holds no data where the adapter expects it.
            MalformedCorpusDataError: If the data is present but violates its own format.
        """
        ...
