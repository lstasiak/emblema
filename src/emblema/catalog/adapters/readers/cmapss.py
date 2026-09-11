import math
from collections.abc import Iterable
from pathlib import Path

from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_description import CorpusDescription
from emblema.catalog.domain.exceptions import CorpusDataNotFoundError, MalformedCorpusDataError
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


class CmapssCorpusReader:
    """Reads the training trajectories of NASA's C-MAPSS turbofan degradation data set.

    The data set comes as four subsets (FD001 to FD004) that differ in operating conditions and
    fault modes. Each ``train_FD00x.txt`` holds one row per engine cycle: unit number, cycle,
    three operational settings and 21 sensor measurements, separated by spaces. An engine is a
    unit, a sensor is a channel, and a sensor value in a cycle is an observation; cycles are
    equally spaced by construction, so the regime is regular. The operational settings are
    inputs that vary per cycle in FD002 and FD004 and are not channels.

    What this reader registers is the pretraining side of the corpus, the training trajectories
    only. The official test trajectories and their remaining-useful-life targets are labelled
    evaluation data and stay outside the Catalog, so a version's checksum is also the proof that
    no test engine took part in pretraining.

    The checksum covers the bytes of the selected files in subset order, whatever order the
    subsets were named in, so it is the provenance of exactly what was read and two readers over
    the same files agree.
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
        chosen = set(subsets)
        unknown = sorted(chosen - set(SUBSETS))
        if unknown:
            raise ValueError(f"unknown C-MAPSS subsets {unknown}; expected some of {SUBSETS}")
        if not chosen:
            raise ValueError("at least one C-MAPSS subset is needed")
        self._files = tuple(root / f"train_{name}.txt" for name in SUBSETS if name in chosen)

    def describe(self) -> CorpusDescription:
        """Validate the selected files and describe them.

        Raises:
            CorpusDataNotFoundError: If a selected file is missing.
            MalformedCorpusDataError: If a row has the wrong width, a value is not a finite
                number, an engine's cycles do not run 1, 2, 3, ... or a file holds no engine.
        """
        contents = [self._bytes_of(path) for path in self._files]
        lengths = [
            length
            for path, content in zip(self._files, contents, strict=True)
            for length in self._cycles_per_engine(path.name, content)
        ]
        return CorpusDescription(
            channel_schema=ChannelSchema(frozenset(SENSORS)),
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=Checksum.of_chunks(contents),
                unit_count=len(lengths),
                observation_count=sum(lengths) * len(SENSORS),
            ),
        )

    @staticmethod
    def _bytes_of(path: Path) -> bytes:
        if not path.is_file():
            raise CorpusDataNotFoundError(f"{path} is missing")
        return path.read_bytes()

    @staticmethod
    def _cycles_per_engine(name: str, content: bytes) -> list[int]:
        cycles: dict[int, int] = {}
        width = LEADING_COLUMNS + len(SENSORS)
        for number, line in enumerate(content.decode("ascii", errors="replace").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != width:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: expected {width} columns, got {len(fields)}"
                )
            try:
                unit, cycle = int(fields[0]), int(fields[1])
                values = [float(field) for field in fields[2:]]
            except ValueError as error:
                raise MalformedCorpusDataError(f"{name}, line {number}: {error}") from error
            if not all(math.isfinite(value) for value in values):
                raise MalformedCorpusDataError(f"{name}, line {number}: non-finite value")
            expected = cycles.get(unit, 0) + 1
            if cycle != expected:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: engine {unit} jumps to cycle {cycle} after "
                    f"cycle {expected - 1}"
                )
            cycles[unit] = cycle
        if not cycles:
            raise MalformedCorpusDataError(f"{name}: no engine")
        return list(cycles.values())
