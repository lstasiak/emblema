import zipfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

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

_CHANNELS_FOLDER = "channels"
_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_HOUR = 3_600_000_000_000.0
# The origin of every mission's time axis: midnight UTC of the day the anonymised missions start,
# stated rather than read off the data so that whoever answers for the telemetry's events places
# an instant on the same axis without unpickling a channel.
ORIGIN_SECONDS = 946_684_800
ORIGIN_TEXT = "2000-01-01T00:00:00Z"
# Files are hashed in pieces of this size, so a channel of a hundred megabytes is never read whole.
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class EsaAdRawPolicy:
    """Which channels of each mission are read, and the most readings a channel may keep.

    A bytes checksum cannot tell two selections of one file set apart, so the policy's statement
    is the first thing the checksum covers: another selection is another version of the data.

    Attributes:
        channels: Channel numbers per mission, in the order the checksum covers them.
        cap: The most readings a channel keeps; one above it keeps the earliest reading of every
            bin of the smallest whole number of seconds that brings it under.
    """

    channels: Mapping[str, tuple[int, ...]]
    cap: int

    def __post_init__(self) -> None:
        if self.cap < 1:
            raise ValueError(f"a channel keeps at least one reading, got a cap of {self.cap}")
        for mission, numbers in self.channels.items():
            if not numbers or list(numbers) != sorted(set(numbers)):
                raise ValueError(f"{mission}: channels must be distinct and in order")

    def statement(self) -> bytes:
        """The policy in words, the bytes the checksum opens with."""
        selected = "; ".join(
            f"{mission} {','.join(str(number) for number in numbers)}"
            for mission, numbers in self.channels.items()
        )
        return (
            f"esa_ad_raw: channels {selected}; cap {self.cap} readings per channel; "
            f"origin {ORIGIN_TEXT}\n"
        ).encode()


class EsaAdRawCorpusReader:
    """Reads the ESA Anomaly Dataset's telemetry as the satellites sent it, over each whole mission.

    A mission is a directory of ``channels/channel_<n>.zip``, one pickled pandas data frame per
    channel: instants as the index, the values as its one column. Every reading of a selected
    channel is kept at its own instant, so the packets a satellite groups its channels into stay
    apart in time and the spacing inside a channel stays what it was; a channel with more
    readings than the policy's cap is thinned to the earliest reading of every bin of the
    smallest whole number of seconds that brings it under, and no channel of the policy in force
    reaches the cap. The whole span of a mission is read, so a task over its later months can
    hold them out of a pretraining over its earlier ones.

    The policy in force reads the target channels of three packets and one group of Mission1
    (every target channel but those of three sparse packets) and the benchmark's lightweight
    channels of Mission2, whose other channels all share one packet. Units are the calendar months
    of a mission's span, keyed ``<mission>/<YYYY-MM>``; the mission stays the independent unit.
    Channels are named by mission. Observations are placed in hours since midnight UTC of the
    day the anonymised missions start, on one axis for every mission.

    The checksum covers the policy's statement and then the selected missions' channel archives
    in mission order and channel-number order. Unpickling runs the stream it reads, so the files
    are trusted as far as the fetch script's checksums vouch for them. Every mission read is
    kept in memory, since describing and then reading a publication would otherwise unpickle
    each one twice; under the policy in force that is about seven gigabytes for both.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("ESA-Mission1", "ESA-Mission2")
    POLICY: ClassVar[EsaAdRawPolicy] = EsaAdRawPolicy(
        channels={
            "ESA-Mission1": (*range(12, 53), *range(70, 77)),
            "ESA-Mission2": tuple(range(18, 29)),
        },
        cap=20_000_000,
    )

    def __init__(
        self,
        root: Path,
        subsets: Iterable[str] = SUBSETS,
        policy: EsaAdRawPolicy = POLICY,
    ) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "ESA mission")
        unknown = sorted(set(self._subsets) - set(policy.channels))
        if unknown:
            raise ValueError(f"the policy selects no channel of {unknown}")
        self._policy = policy
        self._loaded: dict[str, _Mission] = {}

    def describe(self) -> CorpusDescription:
        units = 0
        observations = 0

        def contents() -> Iterator[bytes]:
            nonlocal units, observations
            yield self._policy.statement()
            for name in self._subsets:
                series = []
                for number, path in self._channel_paths(name):
                    yield from self._chunks_of(path)
                    series.append((number, self._series_of(path)))
                mission = _Mission.assemble(name, series, self._policy.cap)
                self._loaded[name] = mission
                units += len(mission.segments)
                observations += mission.observation_count

        # The checksum consumes the files a chunk at a time, so no archive is ever held whole.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(
                frozenset(
                    Channel(_channel_name(name, number))
                    for name in self._subsets
                    for number in self._policy.channels[name]
                )
            ),
            sampling_regime=SamplingRegime.IRREGULAR,
            content=CorpusContent(
                checksum=checksum, unit_count=units, observation_count=observations
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for name in self._subsets:
            mission = self._mission(name)
            for segment in mission.segments:
                yield CorpusUnit(
                    segment.key, TimeExtent(_hours(segment.start), _hours(segment.end))
                )

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        name = unit.part
        if name is None or name not in self._subsets:
            raise UnknownUnitError(f"{unit} is not a month of a selected mission")
        mission = self._mission(name)
        segment = mission.segment_keyed(unit)
        if segment is None:
            raise UnknownUnitError(f"{name} has no month {unit} in its span")
        return mission.observations_of(segment)

    def _mission(self, name: str) -> "_Mission":
        """The mission, assembled from its archives unless it was read before."""
        if name not in self._loaded:
            self._loaded[name] = _Mission.assemble(
                name,
                [(number, self._series_of(path)) for number, path in self._channel_paths(name)],
                self._policy.cap,
            )
        return self._loaded[name]

    def _channel_paths(self, mission: str) -> list[tuple[int, Path]]:
        """The selected channel archives of a mission in channel-number order."""
        folder = self._root / mission / _CHANNELS_FOLDER
        if not folder.is_dir():
            raise CorpusDataNotFoundError(f"{folder} is missing")
        paths = [
            (number, folder / f"channel_{number}.zip") for number in self._policy.channels[mission]
        ]
        for _, path in paths:
            if not path.is_file():
                raise CorpusDataNotFoundError(f"{path} is missing")
        return paths

    @staticmethod
    def _chunks_of(path: Path) -> Iterator[bytes]:
        with path.open("rb") as archive:
            while chunk := archive.read(_CHUNK_BYTES):
                yield chunk

    @staticmethod
    def _series_of(path: Path) -> "_Series":
        """The instants and values of one channel archive, validated against its format."""
        # Imported here: the extra that provides pandas is needed by this reader alone, and a
        # process publishing another corpus must not need it.
        import pandas as pd

        name = path.name
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                if len(members) != 1:
                    raise MalformedCorpusDataError(
                        f"{name}: expected an archive of one member, got {members}"
                    )
                with archive.open(members[0]) as member:
                    frame = pd.read_pickle(member)
        except zipfile.BadZipFile as error:
            raise MalformedCorpusDataError(f"{name}: not a zip archive") from error
        except MalformedCorpusDataError:
            raise
        # A pickle is a program that builds its object, so a stream that is not the expected one
        # fails wherever it was, with whatever error the half-built object raised.
        except Exception as error:
            raise MalformedCorpusDataError(f"{name}: not a pickled data frame: {error}") from error
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
            raise MalformedCorpusDataError(
                f"{name}: expected a data frame of one column, got {type(frame).__name__}"
                + (f" of {frame.shape[1]} columns" if isinstance(frame, pd.DataFrame) else "")
            )
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise MalformedCorpusDataError(
                f"{name}: expected instants as the index, got {type(frame.index).__name__}"
            )
        if frame.index.hasnans:
            raise MalformedCorpusDataError(f"{name}: an instant is missing")
        times = frame.index.values.astype("datetime64[ns]").astype(np.int64)
        if np.any(times[1:] < times[:-1]):
            raise MalformedCorpusDataError(f"{name}: goes back in time")
        if len(times) and int(times[0]) < ORIGIN_SECONDS * _NANOSECONDS_PER_SECOND:
            raise MalformedCorpusDataError(f"{name}: an instant precedes {ORIGIN_TEXT}")
        values = frame.iloc[:, 0].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise MalformedCorpusDataError(f"{name}: values are {values.dtype}, not numbers")
        if not np.all(np.isfinite(values)):
            raise MalformedCorpusDataError(f"{name}: non-finite value")
        return _Series(times, values)


def _channel_name(mission: str, number: int) -> str:
    return f"{mission}/channel_{number}"


def _hours(instant: int) -> float:
    """An instant on the shared axis, by the arithmetic that places the observations too."""
    return float(
        np.int64(instant - ORIGIN_SECONDS * _NANOSECONDS_PER_SECOND) / _NANOSECONDS_PER_HOUR
    )


@dataclass(frozen=True)
class _Series:
    """One channel as read: instants in nanoseconds since the epoch, sorted, and their values."""

    times: NDArray[np.int64]
    values: NDArray[np.generic]

    def kept_under(self, cap: int) -> "_Series":
        """This channel, thinned only if it holds more readings than ``cap``.

        The earliest reading of every bin of the smallest whole number of seconds that leaves at
        most ``cap`` of them; a bin of one second changes nothing for a channel read no faster.
        """
        if len(self.times) <= cap:
            return self
        seconds = 1
        while True:
            bins = self.times // (seconds * _NANOSECONDS_PER_SECOND)
            first = np.ones(len(bins), dtype=bool)
            first[1:] = bins[1:] != bins[:-1]
            if int(first.sum()) <= cap:
                return _Series(self.times[first], self.values[first])
            seconds += 1


@dataclass(frozen=True)
class _Segment:
    """One calendar month of a mission's span, as nanoseconds since the epoch."""

    key: UnitKey
    start: int
    end: int


@dataclass(frozen=True)
class _Mission:
    """A mission's whole span: the arrays every month of it is served from.

    ``start`` is the whole second of the mission's first kept observation and ``end`` the
    second after its last, both in nanoseconds since the epoch.
    """

    name: str
    start: int
    end: int
    channels: tuple[tuple[str, _Series], ...]
    segments: tuple[_Segment, ...]

    @classmethod
    def assemble(cls, name: str, series: Iterable[tuple[int, _Series]], cap: int) -> "_Mission":
        """Thin the channels under the cap and lay the months over the span they cover."""
        kept = [(number, channel.kept_under(cap)) for number, channel in series]
        spanned = [channel for _, channel in kept if len(channel.times)]
        if not spanned:
            raise MalformedCorpusDataError(f"{name}: no observation in any selected channel")
        first = min(int(channel.times[0]) for channel in spanned) // _NANOSECONDS_PER_SECOND
        last = max(int(channel.times[-1]) for channel in spanned) // _NANOSECONDS_PER_SECOND
        start = first * _NANOSECONDS_PER_SECOND
        end = (last + 1) * _NANOSECONDS_PER_SECOND
        return cls(
            name=name,
            start=start,
            end=end,
            channels=tuple((_channel_name(name, number), channel) for number, channel in kept),
            segments=cls._months(name, start, end),
        )

    @staticmethod
    def _months(name: str, start: int, end: int) -> tuple[_Segment, ...]:
        first = np.datetime64(start, "ns").astype("datetime64[M]")
        last = np.datetime64(end - 1, "ns").astype("datetime64[M]")
        beyond = last + np.timedelta64(1, "M")
        months = np.arange(first, beyond)
        edges = np.arange(first, beyond + np.timedelta64(1, "M")).astype("datetime64[ns]")
        return tuple(
            _Segment(UnitKey.within(name, str(month)), max(int(lower), start), min(int(upper), end))
            for month, lower, upper in zip(
                months, edges[:-1].astype(np.int64), edges[1:].astype(np.int64), strict=True
            )
        )

    @property
    def observation_count(self) -> int:
        return sum(len(channel.times) for _, channel in self.channels)

    def segment_keyed(self, key: UnitKey) -> _Segment | None:
        return next((segment for segment in self.segments if segment.key == key), None)

    def observations_of(self, segment: _Segment) -> Iterator[Observation]:
        """The month's observations of every channel, merged by instant, channels in order."""
        times = []
        values = []
        sources = []
        for index, (_, channel) in enumerate(self.channels):
            lower, upper = np.searchsorted(channel.times, [segment.start, segment.end], side="left")
            times.append(channel.times[lower:upper])
            values.append(channel.values[lower:upper].astype(np.float64))
            sources.append(np.full(upper - lower, index, dtype=np.intp))
        instants = np.concatenate(times)
        order = np.argsort(instants, kind="stable")
        origin = ORIGIN_SECONDS * _NANOSECONDS_PER_SECOND
        hours = ((instants[order] - origin) / _NANOSECONDS_PER_HOUR).tolist()
        names = [self.channels[index][0] for index in np.concatenate(sources)[order].tolist()]
        for name, time, value in zip(
            names, hours, np.concatenate(values)[order].tolist(), strict=True
        ):
            yield Observation(name, time, value)
