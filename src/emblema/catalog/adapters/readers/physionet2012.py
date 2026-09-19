from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
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
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

_SEPARATOR = ","
_HEADER = ("Time", "Parameter", "Value")
_ADMISSION = "00:00"
# What a descriptor says when the value was not recorded.
_NOT_RECORDED = "-1"
_MINUTES_PER_HOUR = 60
# A stamp names the minute it starts, so the protocol's last stamp, 48:00, lies inside an extent
# one minute longer than the 48 hours it frames.
_STAY = TimeExtent(0.0, 48.0 + 1.0 / _MINUTES_PER_HOUR)


@dataclass(frozen=True)
class _Stay:
    """One stay as read: its descriptors as features, its measurements as observations."""

    static_features: tuple[StaticFeature, ...]
    observations: tuple[Observation, ...]


class Physionet2012CorpusReader:
    """Reads the ICU stays of the PhysioNet/CinC Challenge 2012, the irregular clinical corpus.

    A stay is one ``set-<x>/<RecordID>.txt`` of ``Time,Parameter,Value`` rows: at ``00:00`` the
    general descriptors, then up to 37 clinical variables at whatever minutes they were measured
    over the first 48 hours. A stay is a unit and a variable a channel; the regime promises no
    cadence, and a stay may hold descriptors and nothing else. The extent is the protocol's, the
    same for every stay, so a window of the whole stay is one window per unit.

    The descriptors become timeless tokens: age, height and gender as the numbers the files carry,
    the ICU type as a presence token of one of four ward channels, because the wards have no order
    a z-score could respect; a descriptor recorded as ``-1`` is no feature. The weight at admission
    is the first measurement of the ``Weight`` series rather than a descriptor of its own: it is a
    weight, taken at hour zero, and the series carries the later ones. The record identifier is
    the unit's key and nothing else.

    Only sets A and B are read. Set C is the challenge's test set and a downstream task's frozen
    side, which never enters a corpus a backbone is pretrained on. The checksum covers the selected
    sets' files in set order and, within a set, in file-name order.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("set-a", "set-b")
    # The 37 clinical variables the challenge describes, with the units it states for them.
    SERIES: ClassVar[tuple[Channel, ...]] = (
        Channel("ALP", "IU/L"),
        Channel("ALT", "IU/L"),
        Channel("AST", "IU/L"),
        Channel("Albumin", "g/dL"),
        Channel("BUN", "mg/dL"),
        Channel("Bilirubin", "mg/dL"),
        Channel("Cholesterol", "mg/dL"),
        Channel("Creatinine", "mg/dL"),
        Channel("DiasABP", "mmHg"),
        Channel("FiO2"),
        Channel("GCS"),
        Channel("Glucose", "mg/dL"),
        Channel("HCO3", "mmol/L"),
        Channel("HCT", "%"),
        Channel("HR", "bpm"),
        Channel("K", "mEq/L"),
        Channel("Lactate", "mmol/L"),
        Channel("MAP", "mmHg"),
        Channel("MechVent"),
        Channel("Mg", "mmol/L"),
        Channel("NIDiasABP", "mmHg"),
        Channel("NIMAP", "mmHg"),
        Channel("NISysABP", "mmHg"),
        Channel("Na", "mEq/L"),
        Channel("PaCO2", "mmHg"),
        Channel("PaO2", "mmHg"),
        Channel("Platelets", "cells/nL"),
        Channel("RespRate", "bpm"),
        Channel("SaO2", "%"),
        Channel("SysABP", "mmHg"),
        Channel("Temp", "°C"),
        Channel("TroponinI", "μg/L"),
        Channel("TroponinT", "μg/L"),
        Channel("Urine", "mL"),
        Channel("WBC", "cells/nL"),
        Channel("Weight", "kg"),
        Channel("pH"),
    )
    # Descriptors kept as the numbers the files carry; the ward is a code and is mapped below.
    MEASURED_DESCRIPTORS: ClassVar[Mapping[str, Channel]] = {
        "Age": Channel("Age", "years", timeless=True),
        "Gender": Channel("Gender", timeless=True),
        "Height": Channel("Height", "cm", timeless=True),
    }
    # The ward a stay was in, by the code the files give it; a stay carries the token of its own.
    WARDS: ClassVar[Mapping[str, Channel]] = {
        "1": Channel("ICUType/coronary_care", timeless=True),
        "2": Channel("ICUType/cardiac_surgery_recovery", timeless=True),
        "3": Channel("ICUType/medical", timeless=True),
        "4": Channel("ICUType/surgical", timeless=True),
    }
    SCHEMA: ClassVar[ChannelSchema] = ChannelSchema(
        frozenset((*SERIES, *MEASURED_DESCRIPTORS.values(), *WARDS.values()))
    )
    _RECORD_ID: ClassVar[str] = "RecordID"
    _WARD: ClassVar[str] = "ICUType"
    _WEIGHT: ClassVar[str] = "Weight"
    _DESCRIPTORS: ClassVar[frozenset[str]] = frozenset((_RECORD_ID, _WARD, *MEASURED_DESCRIPTORS))
    _VARIABLES: ClassVar[frozenset[str]] = frozenset(channel.name for channel in SERIES)

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "PhysioNet set")

    def describe(self) -> CorpusDescription:
        counts: list[int] = []

        def contents() -> Iterator[bytes]:
            for path in self._paths():
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
        for path in self._paths():
            stay = self._stay_of(path, path.read_bytes())
            yield CorpusUnit(self._key_of(path), _STAY, stay.static_features)

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        path = self._locate(unit)
        return iter(self._stay_of(path, path.read_bytes()).observations)

    def _paths(self) -> Iterator[Path]:
        """The selected files in canonical order: sets as declared, stays by name within one."""
        for subset in self._subsets:
            folder = self._root / subset
            if not folder.is_dir():
                raise CorpusDataNotFoundError(f"{folder} is missing")
            files = sorted(folder.glob("*.txt"), key=lambda path: path.name)
            if not files:
                raise CorpusDataNotFoundError(f"{folder} holds no stay")
            yield from files

    def _key_of(self, path: Path) -> UnitKey:
        return UnitKey.within(path.parent.name, path.stem)

    def _locate(self, unit: UnitKey) -> Path:
        subset = unit.part
        if subset is None or subset not in self._subsets:
            raise UnknownUnitError(f"{unit} is not a stay of a selected set")
        folder = self._root / subset
        if not folder.is_dir():
            raise CorpusDataNotFoundError(f"{folder} is missing")
        path = folder / f"{unit.name}.txt"
        # A key naming anything but a file of the set — a nested path, a step upwards — names no
        # stay, however the file system would resolve it.
        if path.parent != folder or not path.is_file():
            raise UnknownUnitError(f"{folder} has no stay {unit.name}")
        return path

    @classmethod
    def _stay_of(cls, path: Path, content: bytes) -> _Stay:
        """The stay ``content`` holds, checked against the file it was read from.

        Raises:
            MalformedCorpusDataError: If the file is not a stay of the challenge's format, names
                another record than its file, or measures anything outside the 48-hour protocol.
        """
        name = f"{path.parent.name}/{path.name}"
        record: str | None = None
        # Which descriptors the file carried, not which became features: one recorded as -1 is no
        # feature, and a file that then gave it a value would repeat it unnoticed.
        given: set[str] = set()
        features: dict[str, StaticFeature] = {}
        observations: list[Observation] = []
        last = 0.0
        for number, stamp, parameter, value in cls._rows_of(name, content):
            if parameter in cls._DESCRIPTORS:
                if stamp != _ADMISSION:
                    raise MalformedCorpusDataError(
                        f"{name}, line {number}: descriptor {parameter} away from admission"
                    )
                if parameter == cls._RECORD_ID:
                    if record is not None:
                        raise MalformedCorpusDataError(f"{name}, line {number}: a second RecordID")
                    if value != path.stem:
                        raise MalformedCorpusDataError(
                            f"{name}, line {number}: names record {value}, not {path.stem}"
                        )
                    record = value
                    continue
                if parameter in given:
                    raise MalformedCorpusDataError(
                        f"{name}, line {number}: descriptor {parameter} given twice"
                    )
                given.add(parameter)
                feature = cls._feature_of(name, number, parameter, value)
                if feature is not None:
                    features[parameter] = feature
                continue
            if parameter not in cls._VARIABLES:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: unknown parameter {parameter}"
                )
            time = cls._hours_of(name, number, stamp)
            if time < last:
                raise MalformedCorpusDataError(f"{name}, line {number}: goes back in time")
            last = time
            if parameter == cls._WEIGHT and stamp == _ADMISSION and value == _NOT_RECORDED:
                continue
            observations.append(
                Observation(parameter, time, finite_floats(name, number, [value])[0])
            )
        if record is None:
            raise MalformedCorpusDataError(f"{name}: no RecordID")
        return _Stay(tuple(features.values()), tuple(observations))

    @staticmethod
    def _rows_of(name: str, content: bytes) -> Iterator[tuple[int, str, str, str]]:
        """Every row after the header, as its line number, stamp, parameter and value.

        Raises:
            MalformedCorpusDataError: If the header is missing or not the challenge's, or a row
                holds another number of fields.
        """
        lines = numbered_lines(content.split(b"\n"))
        first = next(lines, None)
        if first is None:
            raise MalformedCorpusDataError(f"{name}: no header")
        number, header = first
        fields = tuple(fields_of(name, number, header, _SEPARATOR, len(_HEADER)))
        if fields != _HEADER:
            raise MalformedCorpusDataError(
                f"{name}, line {number}: expected the header {_HEADER}, got {fields}"
            )
        for number, line in lines:
            stamp, parameter, value = fields_of(name, number, line, _SEPARATOR, len(_HEADER))
            yield number, stamp, parameter, value

    @classmethod
    def _feature_of(
        cls, name: str, number: int, parameter: str, value: str
    ) -> StaticFeature | None:
        """The timeless token a descriptor becomes; ``None`` where it was not recorded.

        Raises:
            MalformedCorpusDataError: If a ward code is not one of the four, or a value is not a
                finite number.
        """
        if value == _NOT_RECORDED:
            return None
        if parameter == cls._WARD:
            ward = cls.WARDS.get(value)
            if ward is None:
                raise MalformedCorpusDataError(
                    f"{name}, line {number}: no ward has the code {value}; the codes are "
                    f"{sorted(cls.WARDS)}"
                )
            return StaticFeature(ward.name, 1.0)
        channel = cls.MEASURED_DESCRIPTORS[parameter]
        return StaticFeature(channel.name, finite_floats(name, number, [value])[0])

    @staticmethod
    def _hours_of(name: str, number: int, stamp: str) -> float:
        """The instant an ``HH:MM`` stamp names, in hours since admission, inside the protocol.

        Raises:
            MalformedCorpusDataError: If the stamp is not two numbers around a colon, names a
                minute past 59, or lies past the protocol's 48 hours.
        """
        hours, colon, minutes = stamp.partition(":")
        # A digit ``int`` refuses is still a digit to ``isdigit`` — a superscript two, a numeral of
        # another script — so the stamp has to be ASCII before it is read as two numbers.
        if not (
            colon
            and stamp.isascii()
            and hours.isdigit()
            and minutes.isdigit()
            and int(minutes) < _MINUTES_PER_HOUR
        ):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: {stamp!r} is not an HH:MM stamp"
            )
        time = int(hours) + int(minutes) / _MINUTES_PER_HOUR
        if not _STAY.contains(time):
            raise MalformedCorpusDataError(
                f"{name}, line {number}: {stamp} lies past the 48-hour protocol"
            )
        return time
