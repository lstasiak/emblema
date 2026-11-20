from collections.abc import Collection, Iterable, Mapping

from emblema.catalog.domain.exceptions import CorpusDataNotFoundError, InvalidUnitKeyError
from emblema.catalog.domain.identifiers import UnitKey


class ExcludedMonths:
    """The months cut from a corpus by name, each keyed ``<part>/<YYYY-MM>`` within a selected part.

    A corpus whose units are the calendar months of a few long systems (a satellite mission, a
    turbine) is cut by naming months: the months a downstream task reads, or the ones held unread.
    Another cut of the same files is another version of the data, so the months named are part of
    what its checksum covers; a corpus cut by nothing keeps the checksum it always had.
    """

    def __init__(self, names: Iterable[str], parts: Collection[str], part: str) -> None:
        """Take the months named, each within one of the ``parts`` selected.

        Args:
            names: The months, as their units are keyed.
            parts: The selected parts of the corpus.
            part: What one part of this corpus is called, for the refusals.

        Raises:
            ValueError: If a name is no unit key, or is not keyed within a selected part.
        """
        self._part = part
        self._keys = frozenset(self._keyed(name, parts) for name in names)

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def statement(self) -> bytes:
        """The months in words, for the checksum; nothing where none is named."""
        if not self._keys:
            return b""
        return f"excluded {','.join(sorted(map(str, self._keys)))}\n".encode()

    def kept[Month](self, part: str, months: Mapping[UnitKey, Month]) -> tuple[Month, ...]:
        """The months of one part, by their keys, less those excluded, in the order given.

        Raises:
            CorpusDataNotFoundError: If an excluded month of the part is none of ``months``: an
                exclusion that silently missed would let a month meant to stay out be read.
        """
        missed = sorted(str(key) for key in self._keys if key.part == part and key not in months)
        if missed:
            raise CorpusDataNotFoundError(f"{self._part} {part} has no month {missed} to exclude")
        return tuple(month for key, month in months.items() if key not in self._keys)

    def _keyed(self, name: str, parts: Collection[str]) -> UnitKey:
        try:
            key = UnitKey(name)
        except InvalidUnitKeyError as error:
            raise ValueError(f"an excluded month must be named, got {name!r}") from error
        if key.part not in parts:
            raise ValueError(
                f"{name!r} is not keyed <{self._part}>/<YYYY-MM> within a selected {self._part}"
            )
        return key
