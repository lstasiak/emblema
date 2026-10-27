import csv
import gzip
import io
import math
import re
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    InvalidUnitKeyError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

_MINUTES_PER_HOUR = 60
_LAST_MINUTE = 48 * _MINUTES_PER_HOUR
# The first 48 hours on the unit, as the 2012 challenge frames a stay: an offset names the minute
# it starts, so the last one, 2880, lies inside an extent one minute longer.
_STAY = TimeExtent(0.0, 48.0 + 1.0 / _MINUTES_PER_HOUR)
# The age the database writes for every patient older than 89, read as the 2012 stays cap it.
_OLDEST = "> 89"
_OLDEST_AGE = 90.0
_GENDERS: Mapping[str, float] = {"Female": 0.0, "Male": 1.0}
# An offset in minutes, in ASCII digits: ``int`` would also take a sign, spaces, underscores and the
# digits of other scripts.
_OFFSET = re.compile(r"-?[0-9]+")
_URINE_OUTPUT = "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"
# Output cells holding a volume of urine name it, a void, or the catheter or tube it drains through.
_URINE = ("urin", "void", "foley", "urethral", "indwelling catheter", "nephrostomy")
# Output cells that name urine but count voids, record one unmeasured, or mix it with stool.
_NOT_A_VOLUME = ("count", "occurrence", "stool", "unmeasured")
_OXYGEN = "FiO2"
# The fraction of inspired oxygen is at least 21 as a percent, so a value of 1 or less is the same
# quantity written as a fraction.
_LARGEST_FRACTION = 1.0
_PERCENT = 100.0
_NURSE_CATEGORIES = frozenset({"Vital Signs", "Scores"})


@dataclass(frozen=True)
class _Table:
    """One table the reader reads, and the columns it reads of it.

    Attributes:
        name: The file under the root, as the database publishes it.
        columns: The columns the reader needs in its header.
    """

    name: str
    columns: tuple[str, ...]


_PATIENT = _Table(
    "patient.csv.gz",
    (
        "patientunitstayid",
        "hospitalid",
        "gender",
        "age",
        "admissionheight",
        "admissionweight",
        "unittype",
        "unitdischargeoffset",
    ),
)
_NURSE = _Table(
    "nurseCharting.csv.gz",
    (
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypecat",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ),
)
_LAB = _Table(
    "lab.csv.gz",
    ("patientunitstayid", "labresultoffset", "labname", "labresult", "labresultrevisedoffset"),
)
_OUTPUT = _Table(
    "intakeOutput.csv.gz",
    ("patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric"),
)
_TABLES = (_PATIENT, _NURSE, _LAB, _OUTPUT)


@dataclass(frozen=True)
class _Reading:
    """One measurement of a stay, and what decides between two of one channel at one minute.

    Attributes:
        channel: The 2012 variable it is read as.
        minute: Minutes since the stay began on the unit.
        value: The number.
        precedence: Lower is kept: a revised laboratory result before one never revised, a later
            revision before an earlier one, then the first row of its table.
    """

    channel: str
    minute: int
    value: float
    precedence: tuple[int, int, int]


@dataclass(frozen=True)
class _Index:
    """What a pass over the tables yields: every stay with its descriptors and observations."""

    units: Mapping[UnitKey, CorpusUnit]
    observations: Mapping[UnitKey, tuple[Observation, ...]]


class EicuDemoCorpusReader:
    """Reads the unit stays of the eICU Collaborative Research Database demo, as 2012 frames one.

    The root holds the database's tables as it publishes them; four are read. A unit stay is a
    unit, keyed ``<hospital>/<stay>``, over 48 hours from admission to the unit, as the 2012
    challenge frames a stay, and its variables are read as the 2012 variables they are (ADR-0066):
    the vital signs and coma score a nurse charted, the laboratory results, and the urine output.
    Readings before admission, after the patient left the unit or after hour 48 are not the stay's,
    so a stay shorter than 48 hours keeps the extent and leaves its tail empty. Every stay of the
    patient table is a unit, one without a reading too.

    A reading is a finite number. Of two readings of one variable at one minute, one is kept: a
    laboratory result's latest revision, otherwise the first row of its table. Values are read as
    published, except that the fraction of inspired oxygen is read in the percent most results
    give it.

    The descriptors become timeless tokens as the 2012 stays' do: age, height and gender as
    numbers, the unit type as a presence token of its own channel; a gender other than female or
    male is no feature, as an unrecorded one is. The weight at admission is the first reading of
    the ``Weight`` series, at minute zero.

    The checksum covers the four tables in a fixed order, and the counts describe the bytes it
    covers. The tables are read once, when the corpus is described or first read.
    """

    # The 2012 variables the demo charts, in the units its laboratory table gives them; the
    # charted ones in the units the 2012 stays state, which the demo does not.
    SERIES: ClassVar[tuple[Channel, ...]] = (
        Channel("ALP", "Units/L"),
        Channel("ALT", "Units/L"),
        Channel("AST", "Units/L"),
        Channel("Albumin", "g/dL"),
        Channel("BUN", "mg/dL"),
        Channel("Bilirubin", "mg/dL"),
        Channel("Cholesterol", "mg/dL"),
        Channel("Creatinine", "mg/dL"),
        Channel("DiasABP", "mmHg"),
        Channel("FiO2", "%"),
        Channel("GCS"),
        Channel("Glucose", "mg/dL"),
        Channel("HCO3", "mmol/L"),
        Channel("HCT", "%"),
        Channel("HR", "bpm"),
        Channel("K", "mmol/L"),
        Channel("Lactate", "mmol/L"),
        Channel("MAP", "mmHg"),
        Channel("Mg", "mg/dL"),
        Channel("NIDiasABP", "mmHg"),
        Channel("NIMAP", "mmHg"),
        Channel("NISysABP", "mmHg"),
        Channel("Na", "mmol/L"),
        Channel("PaCO2", "mm Hg"),
        Channel("PaO2", "mm Hg"),
        Channel("Platelets", "K/mcL"),
        Channel("RespRate", "bpm"),
        Channel("SaO2", "%"),
        Channel("SysABP", "mmHg"),
        Channel("Temp", "°C"),
        Channel("TroponinI", "ng/mL"),
        Channel("TroponinT", "ng/mL"),
        Channel("Urine", "mL"),
        Channel("WBC", "K/mcL"),
        Channel("Weight", "kg"),
        Channel("pH"),
    )
    # The charted cells read, by the name of their value, as the 2012 variable each one is.
    CHARTED: ClassVar[Mapping[str, str]] = {
        "Heart Rate": "HR",
        "Respiratory Rate": "RespRate",
        "Non-Invasive BP Systolic": "NISysABP",
        "Non-Invasive BP Diastolic": "NIDiasABP",
        "Non-Invasive BP Mean": "NIMAP",
        "Invasive BP Systolic": "SysABP",
        "Invasive BP Diastolic": "DiasABP",
        "Invasive BP Mean": "MAP",
        "Temperature (C)": "Temp",
        "GCS Total": "GCS",
    }
    # The laboratory results read, by name, as the 2012 variable each one is; the serum
    # bicarbonate rather than the one a blood gas computes, the arterial saturation rather than
    # the pulse oximeter's.
    LABORATORY: ClassVar[Mapping[str, str]] = {
        "FiO2": "FiO2",
        "pH": "pH",
        "paCO2": "PaCO2",
        "paO2": "PaO2",
        "bicarbonate": "HCO3",
        "sodium": "Na",
        "potassium": "K",
        "magnesium": "Mg",
        "BUN": "BUN",
        "creatinine": "Creatinine",
        "glucose": "Glucose",
        "Hct": "HCT",
        "platelets x 1000": "Platelets",
        "WBC x 1000": "WBC",
        "albumin": "Albumin",
        "alkaline phos.": "ALP",
        "ALT (SGPT)": "ALT",
        "AST (SGOT)": "AST",
        "total bilirubin": "Bilirubin",
        "total cholesterol": "Cholesterol",
        "lactate": "Lactate",
        "troponin - I": "TroponinI",
        "troponin - T": "TroponinT",
        "O2 Sat (%)": "SaO2",
    }
    MEASURED_DESCRIPTORS: ClassVar[tuple[Channel, ...]] = (
        Channel("Age", "years", timeless=True),
        Channel("Gender", timeless=True),
        Channel("Height", "cm", timeless=True),
    )
    # The unit types the database names, each a presence token of its own channel.
    UNIT_TYPES: ClassVar[Mapping[str, Channel]] = {
        name: Channel(f"UnitType/{name}", timeless=True)
        for name in (
            "CCU-CTICU",
            "CSICU",
            "CTICU",
            "Cardiac ICU",
            "MICU",
            "Med-Surg ICU",
            "Neuro ICU",
            "SICU",
        )
    }
    SCHEMA: ClassVar[ChannelSchema] = ChannelSchema(
        frozenset((*SERIES, *MEASURED_DESCRIPTORS, *UNIT_TYPES.values()))
    )
    _WEIGHT: ClassVar[str] = "Weight"

    def __init__(self, root: Path) -> None:
        self._root = root
        self._index: _Index | None = None

    def describe(self) -> CorpusDescription:
        contents = self._contents()
        checksum = Checksum.of_chunks(contents)
        self._index = self._index_of(contents)
        return CorpusDescription(
            channel_schema=self.SCHEMA,
            sampling_regime=SamplingRegime.IRREGULAR,
            content=CorpusContent(
                checksum=checksum,
                unit_count=len(self._index.units),
                observation_count=sum(len(o) for o in self._index.observations.values()),
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        return iter(self._indexed().units.values())

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        observations = self._indexed().observations.get(unit)
        if observations is None:
            raise UnknownUnitError(f"{unit} is not a stay of the demo")
        return iter(observations)

    def _contents(self) -> list[bytes]:
        """The bytes of each table, in the order the checksum covers them."""
        paths = [self._root / table.name for table in _TABLES]
        absent = [path.name for path in paths if not path.is_file()]
        if absent:
            raise CorpusDataNotFoundError(f"no {', '.join(absent)} under {self._root}")
        return [path.read_bytes() for path in paths]

    def _indexed(self) -> _Index:
        if self._index is None:
            self._index = self._index_of(self._contents())
        return self._index

    @classmethod
    def _index_of(cls, contents: Sequence[bytes]) -> _Index:
        """Every stay of the patient table, with its descriptors and its readings in time order.

        Raises:
            MalformedCorpusDataError: If a table names a stay the patient table does not, a stay
                twice, or a stay that leaves the unit before it arrives.
        """
        patients, nursing, laboratory, output = contents
        units: dict[str, CorpusUnit] = {}
        last: dict[str, int] = {}
        weights: dict[str, float] = {}
        for where, _, row in cls._rows_of(patients, _PATIENT):
            stay = row["patientunitstayid"]
            if stay in units:
                raise MalformedCorpusDataError(f"{where}: stay {stay} given twice")
            units[stay] = CorpusUnit(cls._key_of(where, row), _STAY, cls._features_of(where, row))
            discharge = cls._offset_of(where, row["unitdischargeoffset"])
            if discharge < 0:
                raise MalformedCorpusDataError(f"{where}: stay {stay} leaves before it arrives")
            last[stay] = min(discharge, _LAST_MINUTE)
            weight = cls._number(row["admissionweight"])
            if weight is not None:
                weights[stay] = weight
        chosen: dict[tuple[str, str, int], _Reading] = {}
        for stay, reading in (
            *cls._charted(nursing),
            *cls._laboratory(laboratory),
            *cls._urine(output),
        ):
            if stay not in units:
                raise MalformedCorpusDataError(f"a reading of stay {stay}, which no patient has")
            if reading.minute > last[stay]:
                continue
            slot = (stay, reading.channel, reading.minute)
            if slot not in chosen or reading.precedence < chosen[slot].precedence:
                chosen[slot] = reading
        observations: dict[str, list[Observation]] = defaultdict(list)
        for stay, weight in weights.items():
            observations[stay].append(Observation(cls._WEIGHT, 0.0, weight))
        for (stay, _, _), reading in chosen.items():
            observations[stay].append(
                Observation(reading.channel, reading.minute / _MINUTES_PER_HOUR, reading.value)
            )
        ordered = sorted(units.values(), key=lambda unit: str(unit.key))
        return _Index(
            units={unit.key: unit for unit in ordered},
            observations={
                units[stay].key: tuple(
                    sorted(observations.get(stay, []), key=lambda o: (o.time, o.channel))
                )
                for stay in units
            },
        )

    @classmethod
    def _charted(cls, content: bytes) -> Iterator[tuple[str, _Reading]]:
        for where, line, row in cls._rows_of(content, _NURSE):
            channel = cls.CHARTED.get(row["nursingchartcelltypevalname"])
            if channel is None or row["nursingchartcelltypecat"] not in _NURSE_CATEGORIES:
                continue
            reading = cls._reading_of(
                where, line, channel, row["nursingchartoffset"], row["nursingchartvalue"], (0, 0)
            )
            if reading is not None:
                yield row["patientunitstayid"], reading

    @classmethod
    def _laboratory(cls, content: bytes) -> Iterator[tuple[str, _Reading]]:
        for where, line, row in cls._rows_of(content, _LAB):
            channel = cls.LABORATORY.get(row["labname"])
            if channel is None:
                continue
            revised = row["labresultrevisedoffset"]
            revision = (1, 0) if not revised else (0, -cls._offset_of(where, revised))
            reading = cls._reading_of(
                where, line, channel, row["labresultoffset"], row["labresult"], revision
            )
            if reading is None:
                continue
            if channel == _OXYGEN and reading.value <= _LARGEST_FRACTION:
                reading = replace(reading, value=reading.value * _PERCENT)
            yield row["patientunitstayid"], reading

    @classmethod
    def _urine(cls, content: bytes) -> Iterator[tuple[str, _Reading]]:
        for where, line, row in cls._rows_of(content, _OUTPUT):
            label = row["celllabel"].casefold()
            if not row["cellpath"].startswith(_URINE_OUTPUT):
                continue
            if not any(word in label for word in _URINE):
                continue
            if any(word in label for word in _NOT_A_VOLUME):
                continue
            reading = cls._reading_of(
                where, line, "Urine", row["intakeoutputoffset"], row["cellvaluenumeric"], (0, 0)
            )
            if reading is not None:
                yield row["patientunitstayid"], reading

    @classmethod
    def _reading_of(
        cls, where: str, line: int, channel: str, offset: str, value: str, revision: tuple[int, int]
    ) -> _Reading | None:
        """The reading a row is, or ``None`` where it is no number or lies outside the 48 hours.

        Raises:
            MalformedCorpusDataError: If its offset is not a whole number of minutes.
        """
        minute = cls._offset_of(where, offset)
        number = cls._number(value)
        if number is None or not 0 <= minute <= _LAST_MINUTE:
            return None
        return _Reading(channel, minute, number, (*revision, line))

    @staticmethod
    def _offset_of(where: str, text: str) -> int:
        """Minutes since admission to the unit.

        Raises:
            MalformedCorpusDataError: If the text is not a whole number of minutes.
        """
        if not _OFFSET.fullmatch(text):
            raise MalformedCorpusDataError(f"{where}: {text!r} is not an offset in minutes")
        return int(text)

    @staticmethod
    def _number(text: str) -> float | None:
        """A finite number, or ``None`` where the cell holds none: charting is free text."""
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _key_of(where: str, row: Mapping[str, str]) -> UnitKey:
        """The unit a stay is.

        Raises:
            MalformedCorpusDataError: If the row names no hospital or no stay.
        """
        try:
            return UnitKey.within(row["hospitalid"], row["patientunitstayid"])
        except InvalidUnitKeyError as error:
            raise MalformedCorpusDataError(f"{where}: {error}") from error

    @classmethod
    def _features_of(cls, where: str, row: Mapping[str, str]) -> tuple[StaticFeature, ...]:
        """The timeless tokens of a stay's descriptors; a descriptor not recorded is none.

        Raises:
            MalformedCorpusDataError: If the unit type is not one the database names, or the age
                is neither a number nor the database's mark of the oldest.
        """
        age, gender, height = (channel.name for channel in cls.MEASURED_DESCRIPTORS)
        features: list[StaticFeature] = []
        written = row["age"].strip()
        if written == _OLDEST:
            features.append(StaticFeature(age, _OLDEST_AGE))
        elif written:
            years = cls._number(written)
            if years is None:
                raise MalformedCorpusDataError(f"{where}: {written!r} is not an age")
            features.append(StaticFeature(age, years))
        if row["gender"] in _GENDERS:
            features.append(StaticFeature(gender, _GENDERS[row["gender"]]))
        tall = cls._number(row["admissionheight"])
        if tall is not None:
            features.append(StaticFeature(height, tall))
        unit_type = cls.UNIT_TYPES.get(row["unittype"])
        if unit_type is None:
            raise MalformedCorpusDataError(
                f"{where}: no unit type is called {row['unittype']!r}; the types are "
                f"{sorted(cls.UNIT_TYPES)}"
            )
        features.append(StaticFeature(unit_type.name, 1.0))
        return tuple(features)

    @staticmethod
    def _rows_of(content: bytes, table: _Table) -> Iterator[tuple[str, int, dict[str, str]]]:
        """Every row of a gzipped table, with its line and where it sits for an error to name.

        Raises:
            MalformedCorpusDataError: If the file is not a gzipped UTF-8 CSV table holding the
                columns read.
        """
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as member:
                text = io.TextIOWrapper(member, encoding="utf-8-sig", newline="")
                rows = csv.DictReader(text, restval="")
                absent = [c for c in table.columns if c not in (rows.fieldnames or ())]
                if absent:
                    raise MalformedCorpusDataError(f"{table.name}: no column {absent}")
                for row in rows:
                    yield f"{table.name}, line {rows.line_num}", rows.line_num, row
        except (gzip.BadGzipFile, EOFError, zlib.error) as error:
            raise MalformedCorpusDataError(f"{table.name}: not a gzipped file") from error
        except (UnicodeDecodeError, csv.Error) as error:
            raise MalformedCorpusDataError(
                f"{table.name}: not a UTF-8 CSV table: {error}"
            ) from error
