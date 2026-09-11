from collections.abc import Iterable, Iterator, Sequence

from emblema.catalog.domain.corpus_description import CorpusDescription
from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.observation import Observation


class InMemoryCorpusReader:
    """Reader over data held in memory: hands back the description and units it was given.

    Replacing the data stands in for a corpus that changed between two readings, so the detection
    of modified data can be exercised without any file. The description is taken as given and not
    derived from the units, so a test states what the reader should count.
    """

    def __init__(
        self,
        description: CorpusDescription,
        units: Iterable[tuple[CorpusUnit, Sequence[Observation]]] = (),
    ) -> None:
        """Hold ``description`` and the observations of each unit, in the order given.

        Raises:
            ValueError: If two units share a key.
        """
        self._description = description
        self._units = self._index(units)

    def describe(self) -> CorpusDescription:
        return self._description

    def read_units(self) -> Iterator[CorpusUnit]:
        return (unit for unit, _ in self._units.values())

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        if unit not in self._units:
            raise UnknownUnitError(f"no unit {unit}")
        return iter(self._units[unit][1])

    def replace_data(
        self,
        description: CorpusDescription,
        units: Iterable[tuple[CorpusUnit, Sequence[Observation]]] = (),
    ) -> None:
        self._description = description
        self._units = self._index(units)

    @staticmethod
    def _index(
        units: Iterable[tuple[CorpusUnit, Sequence[Observation]]],
    ) -> dict[UnitKey, tuple[CorpusUnit, tuple[Observation, ...]]]:
        indexed: dict[UnitKey, tuple[CorpusUnit, tuple[Observation, ...]]] = {}
        for unit, observations in units:
            if unit.key in indexed:
                raise ValueError(f"duplicate unit key {unit.key}")
            indexed[unit.key] = (unit, tuple(observations))
        return indexed
