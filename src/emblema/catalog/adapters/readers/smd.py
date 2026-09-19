from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import ClassVar

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.adapters.readers.text import fields_of, finite_floats, numbered_lines
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

_METRIC_COUNT = 38
_SEPARATOR = ","
_PREFIX = "machine"
# The resolution the publishers report. It is nominal: nothing here derives a step from it.
_SAMPLE_PERIOD = 1.0


class SmdCorpusReader:
    """Reads the training halves of the Server Machine Dataset.

    Each ``machine-<group>-<index>.txt`` holds a comma-separated row per minute with 38 metrics
    of one server, unnamed by the publishers. A machine is a unit and a metric a channel; the
    files carry no timestamps, so a row is placed at its index and a machine of ``L`` rows spans
    ``[0, L)``. Only training halves are read: the test halves and their anomaly labels are
    evaluation data outside the Catalog, so a version's checksum also proves that no labelled
    minute was pretrained on.

    The reader is bound to a selection of the three machine groups, each a set of servers of one
    kind and a corpus in its own right. A machine is keyed within its group, as every corpus that
    arrives in parts keys its units, so that a publication can hold a whole group out; the file
    name repeats the group, and the key says it where the other corpora say it. The checksum
    covers the selected groups' files in group order and, within a group, in file-name order,
    whatever order the groups were named in.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("1", "2", "3")
    # The publishers release the metrics unnamed and undocumented, so the channels are named after
    # their column. Zero padding makes the name order the column order, which is the order every
    # consumer of a schema sees.
    METRICS: ClassVar[tuple[Channel, ...]] = tuple(
        Channel(f"metric_{column:02d}") for column in range(1, _METRIC_COUNT + 1)
    )

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "SMD group")

    def describe(self) -> CorpusDescription:
        rows: list[int] = []

        def contents() -> Iterator[bytes]:
            for path in self._paths():
                content = path.read_bytes()
                rows.append(self._rows_in(path.name, content))
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(self.METRICS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(rows),
                observation_count=sum(rows) * len(self.METRICS),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for group, path in self._grouped_paths():
            rows = self._rows_in(path.name, path.read_bytes())
            yield CorpusUnit(
                UnitKey.within(group, path.stem), TimeExtent(0.0, rows * _SAMPLE_PERIOD)
            )

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        path = self._locate(unit)
        return self._observations_of(path.name, path.read_bytes())

    def _paths(self) -> Iterator[Path]:
        """The selected files in canonical order: groups as declared, files by name within one."""
        for _, path in self._grouped_paths():
            yield path

    def _grouped_paths(self) -> Iterator[tuple[str, Path]]:
        """The same files, each beside the group it was read from, which its key states."""
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        found = False
        for group in self._subsets:
            files = sorted(self._root.glob(f"{_PREFIX}-{group}-*.txt"), key=lambda path: path.name)
            found = found or bool(files)
            for path in files:
                yield group, path
        if not found:
            raise CorpusDataNotFoundError(
                f"{self._root} holds no machine of the groups {self._subsets}"
            )

    def _locate(self, unit: UnitKey) -> Path:
        if not self._root.is_dir():
            raise CorpusDataNotFoundError(f"{self._root} is missing")
        group, machine = unit.part, unit.name
        parts = machine.split("-")
        if (
            group is None
            or group not in self._subsets
            or len(parts) != 3
            or parts[0] != _PREFIX
            # The file name names the group too, and a key whose two disagree names no machine.
            or parts[1] != group
        ):
            raise UnknownUnitError(f"{unit} is not a machine of a selected group")
        path = self._root / f"{machine}.txt"
        # A key naming anything but a file of the directory names no machine, however the file
        # system would resolve it.
        if path.parent != self._root or not path.is_file():
            raise UnknownUnitError(f"{self._root} has no machine {machine}")
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
            for metric, value in zip(cls.METRICS, values, strict=True):
                yield Observation(metric.name, minute * _SAMPLE_PERIOD, value)

    @classmethod
    def _rows_of(cls, name: str, content: bytes) -> Iterator[list[float]]:
        for number, line in numbered_lines(content.split(b"\n")):
            yield finite_floats(
                name, number, fields_of(name, number, line, _SEPARATOR, len(cls.METRICS))
            )
