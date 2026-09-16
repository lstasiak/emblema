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

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# The 21 sensor outputs in file column order, named and united as in Saxena, Goebel, Simon and
# Eklund (2008), "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation".
# A ratio has no unit.
SENSORS = (
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

# Unit number, cycle and the three operational settings precede the sensors on every row.
LEADING_COLUMNS = 5
WIDTH = LEADING_COLUMNS + len(SENSORS)


class CmapssCorpusReader:
    """Reads the training trajectories of NASA's C-MAPSS turbofan degradation data set.

    Each ``train_FD00x.txt`` holds a row per engine cycle: unit, cycle, three operational settings
    and 21 sensors. An engine is a unit and a sensor a channel; cycles are evenly spaced, so the
    regime is regular, and the operational settings are not channels. Only training trajectories are
    read: the official test engines and their RUL targets are evaluation data outside the Catalog,
    so a version's checksum also proves that no test engine was pretrained on.

    Units are keyed ``<subset>/<engine>``, since engine numbers restart per subset, and an engine of
    ``L`` cycles spans ``[1, L + 1)``. An engine's rows must form one contiguous block of its file.
    The checksum covers the selected files' bytes in subset order, whatever order the subsets were
    named in.
    """

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        """Bind the reader to the directory holding the ``train_FD00x.txt`` files.

        Args:
            root: Directory the data set was unpacked into.
            subsets: Which of FD001 to FD004 make up the corpus; all four by default. A subset on
                its own is a corpus in its own right, as a single operating regime.

        Raises:
            ValueError: If no subset is named or a name is not one of the four.
        """
        chosen = chosen_subsets(subsets, SUBSETS, "C-MAPSS subset")
        self._files = {name: root / f"train_{name}.txt" for name in chosen}

    def describe(self) -> CorpusDescription:
        """Validate the selected files and describe them.

        Raises:
            CorpusDataNotFoundError: If a selected file is missing.
            MalformedCorpusDataError: If a row has the wrong width, a value is not a finite
                number, an engine's cycles do not run 1, 2, 3, ..., an engine's rows are not one
                contiguous block, or a file holds no engine.
        """
        lengths: list[int] = []

        def contents() -> Iterator[bytes]:
            """The files in subset order, each counted as it passes, then let go."""
            for path in self._files.values():
                content = self._bytes_of(path)
                lengths.extend(self._cycles_per_engine(path.name, content).values())
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(SENSORS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(lengths),
                observation_count=sum(lengths) * len(SENSORS),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        """Every engine of the selected subsets, in file order.

        Raises:
            CorpusDataNotFoundError: If a selected file is missing.
            MalformedCorpusDataError: As for ``describe``.
        """
        for name, path in self._files.items():
            engines = self._cycles_per_engine(path.name, self._bytes_of(path))
            for engine, cycles in engines.items():
                yield CorpusUnit(UnitKey(f"{name}/{engine}"), TimeExtent(1.0, cycles + 1.0))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """The 21 sensor values of every cycle of one engine, cycle by cycle.

        What is known without reading is settled before the stream is handed over; that a file
        holds no such engine is known only once it has been scanned, so that one arrives at the
        end of the stream.

        Raises:
            UnknownUnitError: If the key does not name an engine of a selected subset — when the
                key itself cannot name one, before the stream starts; when the file turns out not
                to hold it, when the stream ends.
            CorpusDataNotFoundError: If the subset's file is missing.
            MalformedCorpusDataError: If a row of the engine is malformed or its cycles do not
                run 1, 2, 3, ...
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
            for number, raw in enumerate(file, 1):
                line = raw.decode("ascii", errors="replace")
                first = line.split(None, 1)
                # A blank line carries nothing, here as in every other scan: it is not the end of
                # the block, and taking it for one would truncate the engine without a word.
                if not first:
                    continue
                if first[0] != marker:
                    if expected > 1:
                        break
                    continue
                _, cycle, sensors = cls._parse_row(path.name, number, line.split())
                if cycle != expected:
                    raise MalformedCorpusDataError(
                        f"{path.name}, line {number}: engine {engine} jumps to cycle {cycle} after "
                        f"cycle {expected - 1}"
                    )
                expected += 1
                for channel, value in zip(SENSORS, sensors, strict=True):
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
        # Lines break on a newline and nothing else, as they do when the file is streamed; str
        # splits on more than that, and the two scans must agree on what a line is.
        lines = content.decode("ascii", errors="replace").split("\n")
        for number, line in enumerate(lines, 1):
            fields = line.split()
            if not fields:
                continue
            engine, cycle, _ = cls._parse_row(name, number, fields)
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

    @staticmethod
    def _parse_row(name: str, number: int, fields: list[str]) -> tuple[int, int, list[float]]:
        if len(fields) != WIDTH:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected {WIDTH} columns, got {len(fields)}"
            )
        try:
            engine, cycle = int(fields[0]), int(fields[1])
            values = [float(field) for field in fields[2:]]
        except ValueError as error:
            raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
        if not all(math.isfinite(value) for value in values):
            raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
        return engine, cycle, values[LEADING_COLUMNS - 2 :]
