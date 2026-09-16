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

# Unit number, cycle and the three operational settings precede the sensors on every row.
_LEADING_COLUMNS = 5
_ENCODING = "ascii"


class CmapssCorpusReader:
    """Reads the training trajectories of NASA's C-MAPSS turbofan degradation data set.

    Each ``train_FD00x.txt`` holds a row per engine cycle: unit, cycle, three operational settings
    and 21 sensors. An engine is a unit and a sensor a channel; cycles are evenly spaced, so the
    regime is regular, and the operational settings are not channels. Only training trajectories are
    read: the official test engines and their RUL targets are evaluation data outside the Catalog,
    so a version's checksum also proves that no test engine was pretrained on.

    The reader is bound to a selection of the four subsets, each one operating regime and a corpus
    in its own right. Units are keyed ``<subset>/<engine>``, since engine numbers restart per
    subset, and an engine of ``L`` cycles spans ``[1, L + 1)``. An engine's rows must form one
    contiguous block of its file. The checksum covers the selected files' bytes in subset order,
    whatever order the subsets were named in.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("FD001", "FD002", "FD003", "FD004")
    # The 21 sensor outputs in file column order, named and united as in Saxena, Goebel, Simon and
    # Eklund (2008), "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
    # A ratio has no unit.
    SENSORS: ClassVar[tuple[Channel, ...]] = (
        Channel("T2", "°R"),
        Channel("T24", "°R"),
        Channel("T30", "°R"),
        Channel("T50", "°R"),
        Channel("P2", "psia"),
        Channel("P15", "psia"),
        Channel("P30", "psia"),
        Channel("Nf", "rpm"),
        Channel("Nc", "rpm"),
        Channel("epr"),
        Channel("Ps30", "psia"),
        Channel("phi", "pps/psi"),
        Channel("NRf", "rpm"),
        Channel("NRc", "rpm"),
        Channel("BPR"),
        Channel("farB"),
        Channel("htBleed"),
        Channel("Nf_dmd", "rpm"),
        Channel("PCNfR_dmd", "rpm"),
        Channel("W31", "lbm/s"),
        Channel("W32", "lbm/s"),
    )
    _WIDTH: ClassVar[int] = _LEADING_COLUMNS + len(SENSORS)

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        chosen = chosen_subsets(subsets, self.SUBSETS, "C-MAPSS subset")
        self._files = {name: root / f"train_{name}.txt" for name in chosen}

    def describe(self) -> CorpusDescription:
        lengths: list[int] = []

        def contents() -> Iterator[bytes]:
            for path in self._files.values():
                content = self._bytes_of(path)
                lengths.extend(self._cycles_per_engine(path.name, content).values())
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(self.SENSORS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(lengths),
                observation_count=sum(lengths) * len(self.SENSORS),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for name, path in self._files.items():
            engines = self._cycles_per_engine(path.name, self._bytes_of(path))
            for engine, cycles in engines.items():
                yield CorpusUnit(UnitKey(f"{name}/{engine}"), TimeExtent(1.0, cycles + 1.0))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """One engine's sensor values, cycle by cycle.

        A key that could name no engine is refused before the stream starts; a file that turns
        out not to hold the engine is reported when the stream ends, because only a scan can tell.
        """
        name, engine = self._locate(unit)
        path = self._files[name]
        if not path.is_file():
            raise CorpusDataNotFoundError(f"{path} is missing")
        return self._observations_of(path, engine)

    @classmethod
    def _observations_of(cls, path: Path, engine: int) -> Iterator[Observation]:
        marker = str(engine)
        expected = 1
        with path.open("rb") as file:
            for number, line in numbered_lines(file, _ENCODING):
                if line.split(None, 1)[0] != marker:
                    if expected > 1:
                        break
                    continue
                _, cycle, sensors = cls._parse_row(path.name, number, line)
                if cycle != expected:
                    raise MalformedCorpusDataError(
                        f"{path.name}, line {number}: engine {engine} jumps to cycle {cycle} after "
                        f"cycle {expected - 1}"
                    )
                expected += 1
                for channel, value in zip(cls.SENSORS, sensors, strict=True):
                    yield Observation(channel.name, float(cycle), value)
        if expected == 1:
            raise UnknownUnitError(f"{path.name} has no engine {engine}")

    def _locate(self, unit: UnitKey) -> tuple[str, int]:
        name, separator, engine = unit.value.partition("/")
        if not separator or name not in self._files or not (engine.isascii() and engine.isdigit()):
            raise UnknownUnitError(f"{unit} is not an engine of a selected subset")
        return name, int(engine)

    @staticmethod
    def _bytes_of(path: Path) -> bytes:
        if not path.is_file():
            raise CorpusDataNotFoundError(f"{path} is missing")
        return path.read_bytes()

    @classmethod
    def _cycles_per_engine(cls, name: str, content: bytes) -> dict[int, int]:
        cycles: dict[int, int] = {}
        current: int | None = None
        for number, line in numbered_lines(content.split(b"\n"), _ENCODING):
            engine, cycle, _ = cls._parse_row(name, number, line)
            if engine != current:
                if engine in cycles:
                    raise MalformedCorpusDataError(
                        f"{name}, line {number}: engine {engine} resumes after engine {current}"
                    )
                current = engine
            expected = cycles.get(engine, 0) + 1
            if cycle != expected:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: engine {engine} jumps to cycle {cycle} after "
                    f"cycle {expected - 1}"
                )
            cycles[engine] = cycle
        if not cycles:
            raise MalformedCorpusDataError(f"{name}: no engine")
        return cycles

    @classmethod
    def _parse_row(cls, name: str, number: int, line: str) -> tuple[int, int, list[float]]:
        fields = fields_of(name, number, line, None, cls._WIDTH)
        try:
            engine, cycle = int(fields[0]), int(fields[1])
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        # The operational settings are held to being numbers like the sensors, and then left out.
        values = finite_floats(name, number, fields[2:])
        return engine, cycle, values[_LEADING_COLUMNS - 2 :]
