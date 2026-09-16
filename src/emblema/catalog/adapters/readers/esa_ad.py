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
# At most one observation per channel is kept in each bin of this many seconds. Mission1's channels
# already run at this spacing; Mission2's run faster and are halved.
_BIN_SECONDS = 30
_BIN_NANOSECONDS = _BIN_SECONDS * _NANOSECONDS_PER_SECOND
_NANOSECONDS_PER_HOUR = 3_600_000_000_000.0
# Files are hashed in pieces of this size, so a channel of a hundred megabytes is never read whole.
_CHUNK_BYTES = 1 << 20


class EsaAdCorpusReader:
    """Reads the training halves of the ESA Anomaly Dataset, the satellite telemetry benchmark.

    A mission is a directory of ``channels/channel_<n>.zip``, one pickled pandas data frame per
    channel: instants as the index, the values as its one column. Only the benchmark's lightweight
    channels are read, only the first half of a mission by time — the benchmark's training
    portion, cut back to a thirty-second bin boundary — and within it the earliest observation of
    each channel in each bin, at its native instant, so the volume is bounded without a grid.

    Units are the calendar months of a mission's training half, keyed ``<mission>/<YYYY-MM>``: the
    two missions are too few to split on and have channels of their own (ADR-0025). A month is a
    unit whether or not any telemetry reached the ground in it. Channels are named
    ``<mission>/channel_<n>``, since a channel number names a different quantity on each
    satellite. Observations are placed in hours since the mission's start, the whole second of
    its first kept observation, on one axis shared by the mission's months; the regime promises
    no cadence.

    The checksum covers the selected missions' channel archives in mission order and
    channel-number order, whatever order the missions were named in. Unpickling runs the stream it
    reads, so the files are trusted as far as the fetch script's checksums vouch for them. The
    mission last read is kept, so that its months are served from memory rather than by
    unpickling a gigabyte each.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("ESA-Mission1", "ESA-Mission2")
    # The channels the benchmark's lightweight configuration monitors, one run of channel numbers
    # per mission: subsystem 5 of Mission1 and subsystem 1 of Mission2. The third published mission
    # is left out as the benchmark leaves it out; adding it is an entry here and a decision on its
    # channels.
    LIGHTWEIGHT: ClassVar[Mapping[str, range]] = {
        "ESA-Mission1": range(41, 47),
        "ESA-Mission2": range(18, 29),
    }

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "ESA mission")
        self._loaded: _Mission | None = None

    def describe(self) -> CorpusDescription:
        units = 0
        observations = 0

        def contents() -> Iterator[bytes]:
            nonlocal units, observations
            for name in self._subsets:
                series = []
                for number, path in self._channel_paths(name):
                    yield from self._chunks_of(path)
                    series.append((number, self._series_of(path)))
                mission = _Mission.assemble(name, series)
                units += len(mission.segments)
                observations += mission.observation_count

        # The checksum consumes the files a chunk at a time, so no archive is ever held whole.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=ChannelSchema(
                frozenset(
                    Channel(_channel_name(name, number))
                    for name in self._subsets
                    for number in self.LIGHTWEIGHT[name]
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
                    segment.key,
                    TimeExtent(mission.hours(segment.start), mission.hours(segment.end)),
                )

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        name, separator, _ = unit.value.partition("/")
        if not separator or name not in self._subsets:
            raise UnknownUnitError(f"{unit} is not a month of a selected mission")
        mission = self._mission(name)
        segment = mission.segment_keyed(unit)
        if segment is None:
            raise UnknownUnitError(f"{name} has no month {unit} in its training half")
        return mission.observations_of(segment)

    def _mission(self, name: str) -> "_Mission":
        """The mission, assembled from its archives unless it is the one read last."""
        if self._loaded is None or self._loaded.name != name:
            self._loaded = _Mission.assemble(
                name,
                [(number, self._series_of(path)) for number, path in self._channel_paths(name)],
            )
        return self._loaded

    def _channel_paths(self, mission: str) -> list[tuple[int, Path]]:
        """The lightweight channel archives of a mission in channel-number order."""
        folder = self._root / mission / _CHANNELS_FOLDER
        if not folder.is_dir():
            raise CorpusDataNotFoundError(f"{folder} is missing")
        paths = [(number, folder / f"channel_{number}.zip") for number in self.LIGHTWEIGHT[mission]]
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
        values = frame.iloc[:, 0].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise MalformedCorpusDataError(f"{name}: values are {values.dtype}, not numbers")
        if not np.all(np.isfinite(values)):
            raise MalformedCorpusDataError(f"{name}: non-finite value")
        return _Series(times, values)


def _channel_name(mission: str, number: int) -> str:
    return f"{mission}/channel_{number}"


@dataclass(frozen=True)
class _Series:
    """One channel as read: instants in nanoseconds since the epoch, sorted, and their values."""

    times: NDArray[np.int64]
    values: NDArray[np.generic]

    def kept_before(self, end: int) -> "_Series":
        """The earliest observation of every bin that starts before ``end``."""
        count = int(np.searchsorted(self.times, end, side="left"))
        times = self.times[:count]
        bins = times // _BIN_NANOSECONDS
        first = np.ones(count, dtype=bool)
        first[1:] = bins[1:] != bins[:-1]
        return _Series(times[first], self.values[:count][first])


@dataclass(frozen=True)
class _Segment:
    """One calendar month of a mission's training half, as nanoseconds since the epoch."""

    key: UnitKey
    start: int
    end: int


@dataclass(frozen=True)
class _Mission:
    """A mission's training half, decimated: the arrays every month of it is served from.

    ``start`` is the origin of the mission's axis, the whole second of its first observation, and
    ``end`` the bin boundary at or before the exact half of the span the selected channels cover,
    both in nanoseconds since the epoch.
    """

    name: str
    start: int
    end: int
    channels: tuple[tuple[str, _Series], ...]
    segments: tuple[_Segment, ...]

    @classmethod
    def assemble(cls, name: str, series: Iterable[tuple[int, _Series]]) -> "_Mission":
        """Cut the channels to the training half and lay the months over it."""
        series = list(series)
        spanned = [channel for _, channel in series if len(channel.times)]
        if not spanned:
            raise MalformedCorpusDataError(f"{name}: no observation in any selected channel")
        first = min(int(channel.times[0]) for channel in spanned) // _NANOSECONDS_PER_SECOND
        last = max(int(channel.times[-1]) for channel in spanned) // _NANOSECONDS_PER_SECOND
        cutoff = first + (last - first) // 2
        end_second = cutoff // _BIN_SECONDS * _BIN_SECONDS
        if end_second <= first:
            raise MalformedCorpusDataError(
                f"{name}: spans {last - first + 1} seconds, too few to hold a training half"
            )
        start = first * _NANOSECONDS_PER_SECOND
        end = end_second * _NANOSECONDS_PER_SECOND
        return cls(
            name=name,
            start=start,
            end=end,
            channels=tuple(
                (_channel_name(name, number), channel.kept_before(end))
                for number, channel in series
            ),
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
            _Segment(UnitKey(f"{name}/{month}"), max(int(lower), start), min(int(upper), end))
            for month, lower, upper in zip(
                months, edges[:-1].astype(np.int64), edges[1:].astype(np.int64), strict=True
            )
        )

    @property
    def observation_count(self) -> int:
        return sum(len(channel.times) for _, channel in self.channels)

    def hours(self, instant: int) -> float:
        """An instant on the mission's axis, by the arithmetic that places the observations too.

        One division for a month's boundary and for an observation on it, so that the two come
        out as the same number and the observation lies inside the month by construction.
        """
        return float(np.int64(instant - self.start) / _NANOSECONDS_PER_HOUR)

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
        hours = ((instants[order] - self.start) / _NANOSECONDS_PER_HOUR).tolist()
        names = [self.channels[index][0] for index in np.concatenate(sources)[order].tolist()]
        for name, time, value in zip(
            names, hours, np.concatenate(values)[order].tolist(), strict=True
        ):
            yield Observation(name, time, value)
