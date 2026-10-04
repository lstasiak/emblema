from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from emblema.catalog.adapters.readers.text import fields_of, finite_floats, numbered_lines
from emblema.catalog.adapters.readers.unit_files import UnitFiles
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import MalformedCorpusDataError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

_SEPARATOR = "|"
_EXTENSION = ".psv"
# What a cell says when nothing was recorded in that hour.
_NOT_RECORDED = "NaN"
_PRESENT = 1.0
_ABSENT = 0.0
_HOUR = "ICULOS"
_LABEL = "SepsisLabel"


@dataclass(frozen=True)
class _Stay:
    """One stay as read: its descriptors as features, its hourly measurements as observations."""

    static_features: tuple[StaticFeature, ...]
    observations: tuple[Observation, ...]
    extent: TimeExtent


class Physionet2019CorpusReader:
    """Reads the ICU stays of the PhysioNet/CinC Challenge 2019, the hourly clinical corpus.

    A stay is one ``training_set<X>/p<nnnnnn>.psv`` of pipe-separated rows, one per hour in the
    unit: 34 clinical variables as measured in that hour or ``NaN``, six descriptors repeated on
    every row, the hour since admission and the challenge's sepsis label. A stay is a unit and a
    variable a channel; most cells are empty, so the regime promises no cadence. Time is the hour
    since admission, and a stay spans from its first recorded hour to the one past its last.

    The descriptors become timeless tokens: age, gender and the hours between hospital and ICU
    admission as numbers, the ward as a presence token of one of two unit channels, since the
    wards have no order a z-score could respect; a descriptor recorded as ``NaN`` is no feature.
    The sepsis label is the downstream task's target: it is checked, never read as a channel.

    The challenge published no test set with labels, so the task's frozen side is cut from these
    sets by name: stays named in ``excluded`` are out of the units, the counts and the checksum,
    and a name matching no stay is refused. The checksum covers the files read, in the order
    ``UnitFiles`` visits them.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("training_setA", "training_setB")
    # The 34 clinical variables of the challenge, with the units it states for them.
    SERIES: ClassVar[tuple[Channel, ...]] = (
        Channel("HR", "bpm"),
        Channel("O2Sat", "%"),
        Channel("Temp", "°C"),
        Channel("SBP", "mmHg"),
        Channel("MAP", "mmHg"),
        Channel("DBP", "mmHg"),
        Channel("Resp", "bpm"),
        Channel("EtCO2", "mmHg"),
        Channel("BaseExcess", "mmol/L"),
        Channel("HCO3", "mmol/L"),
        Channel("FiO2", "%"),
        Channel("pH"),
        Channel("PaCO2", "mmHg"),
        Channel("SaO2", "%"),
        Channel("AST", "IU/L"),
        Channel("BUN", "mg/dL"),
        Channel("Alkalinephos", "IU/L"),
        Channel("Calcium", "mg/dL"),
        Channel("Chloride", "mmol/L"),
        Channel("Creatinine", "mg/dL"),
        Channel("Bilirubin_direct", "mg/dL"),
        Channel("Glucose", "mg/dL"),
        Channel("Lactate", "mg/dL"),
        Channel("Magnesium", "mmol/dL"),
        Channel("Phosphate", "mg/dL"),
        Channel("Potassium", "mmol/L"),
        Channel("Bilirubin_total", "mg/dL"),
        Channel("TroponinI", "ng/mL"),
        Channel("Hct", "%"),
        Channel("Hgb", "g/dL"),
        Channel("PTT", "s"),
        Channel("WBC", "10^3/µL"),
        Channel("Fibrinogen", "mg/dL"),
        Channel("Platelets", "10^3/µL"),
    )
    # Descriptors kept as the numbers the files carry; the wards are flags and are mapped below.
    MEASURED_DESCRIPTORS: ClassVar[Mapping[str, Channel]] = {
        "Age": Channel("Age", "years", timeless=True),
        "Gender": Channel("Gender", timeless=True),
        "HospAdmTime": Channel("HospAdmTime", "hours", timeless=True),
    }
    # The ward a stay was in, by the flag column that marks it; a stay carries the token of its own.
    WARDS: ClassVar[Mapping[str, Channel]] = {
        "Unit1": Channel("Unit/MICU", timeless=True),
        "Unit2": Channel("Unit/SICU", timeless=True),
    }
    SCHEMA: ClassVar[ChannelSchema] = ChannelSchema(
        frozenset((*SERIES, *MEASURED_DESCRIPTORS.values(), *WARDS.values()))
    )
    # The descriptors in the publisher's column order, which puts the wards between the numbers.
    _DESCRIPTORS: ClassVar[tuple[str, ...]] = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
    # The publisher's column order: the variables, the descriptors, the hour, the label.
    HEADER: ClassVar[tuple[str, ...]] = (
        *(channel.name for channel in SERIES),
        *_DESCRIPTORS,
        _HOUR,
        _LABEL,
    )

    def __init__(
        self, root: Path, subsets: Iterable[str] = SUBSETS, excluded: Iterable[str] = ()
    ) -> None:
        self._files = UnitFiles(
            root,
            subsets,
            self.SUBSETS,
            _EXTENSION,
            what="stay",
            part="PhysioNet set",
            excluded=excluded,
        )

    def describe(self) -> CorpusDescription:
        counts: list[int] = []

        def contents() -> Iterator[bytes]:
            for path in self._files.paths():
                content = path.read_bytes()
                counts.append(len(self._stay_of(path, content).observations))
                yield content

        # The checksum consumes the files one at a time, so the whole corpus is never in memory.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=self.SCHEMA,
            sampling_regime=SamplingRegime.IRREGULAR,
            content=CorpusContent(
                checksum=checksum, unit_count=len(counts), observation_count=sum(counts)
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for path in self._files.paths():
            stay = self._stay_of(path, path.read_bytes())
            yield CorpusUnit(self._files.key_of(path), stay.extent, stay.static_features)

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        path = self._files.locate(unit)
        return iter(self._stay_of(path, path.read_bytes()).observations)

    @classmethod
    def _stay_of(cls, path: Path, content: bytes) -> _Stay:
        """The stay ``content`` holds, checked against the challenge's format.

        Raises:
            MalformedCorpusDataError: If the header is not the challenge's, a row holds another
                number of fields or a value that is not a number, the hours do not increase, a
                descriptor changes within the stay, a stay names two wards, or the label is not
                a bit.
        """
        name = f"{path.parent.name}/{path.name}"
        lines = numbered_lines(content.split(b"\n"))
        first = next(lines, None)
        if first is None:
            raise MalformedCorpusDataError(f"{name}: no header")
        number, header = first
        fields = tuple(fields_of(name, number, header, _SEPARATOR, len(cls.HEADER)))
        if fields != cls.HEADER:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected the header {list(cls.HEADER)}, got {list(fields)}"
            )
        descriptors: dict[str, str] | None = None
        described_at = 0
        observations: list[Observation] = []
        first_hour: int | None = None
        last_hour: int | None = None
        timed = len(cls.SERIES)
        for number, line in lines:
            row = fields_of(name, number, line, _SEPARATOR, len(cls.HEADER))
            hour = cls._hour_of(name, number, row[-2], last_hour)
            first_hour = hour if first_hour is None else first_hour
            last_hour = hour
            if row[-1] not in ("0", "1"):
                raise MalformedCorpusDataError(f"{name}, line {number}: {_LABEL} is not a bit")
            given = dict(zip(cls._DESCRIPTORS, row[timed:-2], strict=True))
            if descriptors is None:
                descriptors, described_at = given, number
            elif given != descriptors:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: a descriptor changes within the stay"
                )
            for channel, value in zip(cls.SERIES, row[:timed], strict=True):
                if value != _NOT_RECORDED:
                    observations.append(
                        Observation(
                            channel.name, float(hour), finite_floats(name, number, [value])[0]
                        )
                    )
        if descriptors is None or first_hour is None or last_hour is None:
            raise MalformedCorpusDataError(f"{name}: no row")
        return _Stay(
            cls._features_of(name, described_at, descriptors),
            tuple(observations),
            TimeExtent(float(first_hour), float(last_hour) + 1.0),
        )

    @staticmethod
    def _hour_of(name: str, number: int, cell: str, last: int | None) -> int:
        """The hour a row's ``ICULOS`` names: a whole number after the row before it.

        Raises:
            MalformedCorpusDataError: If the cell is not a whole number or the hour does not
                follow the previous row's.
        """
        if not (cell.isascii() and cell.isdigit()):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: {_HOUR} {cell!r} is not a whole number of hours"
            )
        hour = int(cell)
        if last is not None and hour <= last:
            raise MalformedCorpusDataError(f"{name}, line {number}: goes back in time")
        return hour

    @classmethod
    def _features_of(
        cls, name: str, number: int, descriptors: Mapping[str, str]
    ) -> tuple[StaticFeature, ...]:
        """The timeless tokens the descriptors first given on line ``number`` become.

        Raises:
            MalformedCorpusDataError: If a descriptor is not a number, a ward flag is not a bit,
                or the stay is flagged as in both wards.
        """
        features: list[StaticFeature] = []
        for column, channel in cls.MEASURED_DESCRIPTORS.items():
            value = descriptors[column]
            if value != _NOT_RECORDED:
                features.append(
                    StaticFeature(channel.name, finite_floats(name, number, [value])[0])
                )
        wards: list[Channel] = []
        for column, channel in cls.WARDS.items():
            value = descriptors[column]
            if value == _NOT_RECORDED:
                continue
            flag = finite_floats(name, number, [value])[0]
            if flag not in (_PRESENT, _ABSENT):
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: {column} {value!r} is not a bit"
                )
            if flag == _PRESENT:
                wards.append(channel)
        if len(wards) > 1:
            raise MalformedCorpusDataError(f"{name}, line {number}: a stay in both wards")
        features.extend(StaticFeature(ward.name, _PRESENT) for ward in wards)
        return tuple(features)
