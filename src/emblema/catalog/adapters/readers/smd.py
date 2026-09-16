import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

SUBSETS = ("1", "2", "3")

WIDTH = 38
# The publishers release the metrics unnamed and undocumented, so the channels are named after
# their column. Zero padding makes the name order the column order, which is the order every
# consumer of a schema sees.
METRICS = tuple(Channel(f"metric_{column:02d}") for column in range(1, WIDTH + 1))

SEPARATOR = ","
PREFIX = "machine"
# The publishers report a one-minute resolution and the files carry no timestamps, so a row is a
# minute and a row index is the time. It is a nominal cadence: nothing here derives a step from it.
SAMPLE_PERIOD = 1.0


class SmdCorpusReader:
    """Reads the training halves of the Server Machine Dataset.

    Each ``machine-<group>-<index>.txt`` holds a comma-separated row per minute with 38 metrics
    of one server, unnamed by the publishers. A machine is a unit and a metric a channel; the
    files carry no timestamps, so a row is placed at its index and a machine of ``L`` rows spans
    ``[0, L)``. Only training halves are read: the test halves and their anomaly labels are
    evaluation data outside the Catalog, so a version's checksum also proves that no labelled
    minute was pretrained on.

    Units are keyed by file name, which already names the group. The checksum covers the selected
    groups' files in group order and, within a group, in file-name order, whatever order the
    groups were named in.
    """

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        """Bind the reader to the directory holding the training files.

        Args:
            root: Directory the ``machine-*.txt`` training halves sit in.
            subsets: Which machine groups make up the corpus; all three by default. A group is a
                set of servers of one kind, so on its own it is a corpus in its own right.

        Raises:
            ValueError: If no group is named or a name is not one of the three.
        """
        self._root = root
        self._subsets = chosen_subsets(subsets, SUBSETS, "SMD group")

    def describe(self) -> CorpusDescription:
        """Validate the selected machines and describe them.

        Raises:
            CorpusDataNotFoundError: If the directory is missing or holds no selected machine.
            MalformedCorpusDataError: If a row has the wrong width, a value is not a finite
                number, or a file holds no row.
        """
        rows: list[int] = []

        def contents() -> Iterator[bytes]:
            """The files in canonical order, each counted as it passes, then let go."""
            for path in self._paths():
                content = path.read_bytes()
                rows.append(self._rows_in(path.name, content))
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(METRICS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(rows),
                observation_count=sum(rows) * WIDTH,
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        """Every machine of the selected groups, in canonical order.

        Raises:
            CorpusDataNotFoundError: If the directory is missing or holds no selected machine.
            MalformedCorpusDataError: As for ``describe``.
        """
        for path in self._paths():
            rows = self._rows_in(path.name, path.read_bytes())
            yield CorpusUnit(UnitKey(path.stem), TimeExtent(0.0, rows * SAMPLE_PERIOD))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """The 38 metric values of every minute of one machine, minute by minute.

        Raises:
            UnknownUnitError: If the key does not name a machine of a selected group.
            CorpusDataNotFoundError: If the directory is missing.
            MalformedCorpusDataError: If a row of the machine is malformed.
        """
        path = self._locate(unit)
        return self._observations_of(path.name, path.read_bytes())

    def _paths(self) -> Iterator[Path]:
        """The selected files in canonical order: groups as declared, files by name within one."""
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        found = False
        for group in self._subsets:
            files = sorted(self._root.glob(f"{PREFIX}-{group}-*.txt"), key=lambda path: path.name)
            found = found or bool(files)
            yield from files
        if not found:
            raise CorpusDataNotFoundError(
                f"{self._root} holds no machine of the groups {self._subsets}"
            )

    def _locate(self, unit: UnitKey) -> Path:
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        parts = unit.value.split("-")
        if len(parts) != 3 or parts[0] != PREFIX or parts[1] not in self._subsets:
            raise UnknownUnitError(f"{unit} is not a machine of a selected group")
        path = self._root / f"{unit.value}.txt"
        # A key naming anything but a file of the directory names no machine, however the file
        # system would resolve it.
        if path.parent != self._root or not path.is_file():
            raise UnknownUnitError(f"{self._root} has no machine {unit}")
        return path

    @classmethod
    def _rows_in(cls, name: str, content: bytes) -> int:
        rows = sum(1 for _ in cls._rows_of(name, content))
        if not rows:
            raise MalformedCorpusDataError(f"{name}: no row")
        return rows

    @classmethod
    def _observations_of(cls, name: str, content: bytes) -> Iterator[Observation]:
        for minute, values in enumerate(cls._rows_of(name, content)):
            for metric, value in zip(METRICS, values, strict=True):
                yield Observation(metric.name, minute * SAMPLE_PERIOD, value)

    @classmethod
    def _rows_of(cls, name: str, content: bytes) -> Iterator[list[float]]:
        # Lines break on a newline and nothing else; a carriage return belongs to the line ending,
        # not to the last field.
        lines = content.decode("utf-8", errors="replace").split("\n")
        for number, line in enumerate(lines, 1):
            fields = line.rstrip("\r").split(SEPARATOR)
            # A blank line carries nothing: it is not the end of the machine, and taking it for
            # one would truncate the file without a word.
            if fields == [""]:
                continue
            yield cls._parse_row(name, number, fields)

    @staticmethod
    def _parse_row(name: str, number: int, fields: list[str]) -> list[float]:
        if len(fields) != WIDTH:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected {WIDTH} columns, got {len(fields)}"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        if not all(math.isfinite(value) for value in values):
            raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
        return values
