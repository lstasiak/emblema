import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from itertools import chain

from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channel_vocabulary import VocabularyEntry
from emblema.catalog.domain.corpus_unit import CorpusUnit, TimeExtent
from emblema.catalog.domain.exceptions import (
    ChannelKindMismatchError,
    ObservationOutOfOrderError,
    ObservationOutsideExtentError,
    UnknownChannelError,
)
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.static_feature import StaticFeature
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID, TokenWindow, canonical_key

# A timed observation once normalised: time on the unit's axis, channel id, value. The fields are
# in that order so that sorting the tuples is already the canonical token order, as far as it goes
# without the gap, which is not known until a window is laid.
_NormalisedObservation = tuple[float, int, float]


class SlidingWindowTokeniser:
    """Streams a unit through a window that slides along its time axis, one observation at a time.

    Memory is bounded by the observations one window can hold, never by the unit: a buffer keeps
    the observations of the window currently open, a window is produced when the stream passes its
    end, and observations that fall before the next window's start are dropped. Fitting is a
    separate pass, over the training units rather than one unit, and holds only Welford's running
    moments per channel. So a corpus of any size passes through in one pass per operation.
    """

    def fit(
        self,
        corpus: str,
        observations: Iterable[Observation],
        static_features: Iterable[StaticFeature],
        scheme: TokenisationScheme,
    ) -> TokenisationScheme:
        entries = _entries_by_channel(scheme, corpus)
        moments: dict[int, _RunningMoments] = {}
        for observation in observations:
            entry = _entry(entries, corpus, observation.channel, timeless=False)
            moments.setdefault(entry.channel_id, _RunningMoments()).add(observation.value)
        for feature in static_features:
            entry = _entry(entries, corpus, feature.channel, timeless=True)
            moments.setdefault(entry.channel_id, _RunningMoments()).add(feature.value)
        for channel_id in sorted(moments):
            scheme = scheme.with_statistics(channel_id, moments[channel_id].statistics())
        return scheme

    def tokenise(
        self,
        corpus: str,
        unit: CorpusUnit,
        observations: Iterable[Observation],
        scheme: TokenisationScheme,
        window: WindowSpec,
    ) -> Iterator[PlacedWindow]:
        entries = _entries_by_channel(scheme, corpus)
        statics = self._static_tokens(unit, entries, corpus, scheme)
        extents = window.windows_over(unit.extent)
        current = next(extents, None)
        buffer: deque[_NormalisedObservation] = deque()
        # A time beyond every extent closes the windows the stream itself never reached.
        past_every_window: _NormalisedObservation = (math.inf, PADDING_CHANNEL_ID, 0.0)
        for time, channel_id, value in chain(
            self._normalised_observations(corpus, unit, observations, scheme, entries),
            (past_every_window,),
        ):
            while current is not None and time >= current.end:
                if buffer:
                    tokens = self._token_window(statics, buffer, current)
                    yield PlacedWindow(unit.key, current, tokens)
                current = next(extents, None)
                while current is not None and buffer and buffer[0][0] < current.start:
                    buffer.popleft()
            # With a stride longer than the window an observation can fall between two windows
            # and belongs to neither.
            if current is not None and time >= current.start:
                buffer.append((time, channel_id, value))

    @staticmethod
    def _normalised_observations(
        corpus: str,
        unit: CorpusUnit,
        observations: Iterable[Observation],
        scheme: TokenisationScheme,
        entries: Mapping[str, VocabularyEntry],
    ) -> Iterator[_NormalisedObservation]:
        """The unit's observations, held to what the unit and the scheme say, in arrival order."""
        last_time = -math.inf
        for observation in observations:
            if observation.time < last_time:
                raise ObservationOutOfOrderError(
                    f"unit {unit.key}: observation at {observation.time} follows one at {last_time}"
                )
            last_time = observation.time
            if not unit.extent.contains(observation.time):
                raise ObservationOutsideExtentError(
                    f"unit {unit.key}: observation at {observation.time} lies outside "
                    f"[{unit.extent.start}, {unit.extent.end})"
                )
            entry = _entry(entries, corpus, observation.channel, timeless=False)
            statistics = scheme.statistics_of(entry.channel_id)
            yield observation.time, entry.channel_id, statistics.normalise(observation.value)

    @staticmethod
    def _static_tokens(
        unit: CorpusUnit,
        entries: Mapping[str, VocabularyEntry],
        corpus: str,
        scheme: TokenisationScheme,
    ) -> tuple[tuple[int, float], ...]:
        tokens = []
        for feature in unit.static_features:
            entry = _entry(entries, corpus, feature.channel, timeless=True)
            tokens.append(
                (entry.channel_id, scheme.statistics_of(entry.channel_id).normalise(feature.value))
            )
        return tuple(sorted(tokens))

    @staticmethod
    def _token_window(
        statics: tuple[tuple[int, float], ...],
        buffer: Iterable[_NormalisedObservation],
        extent: TimeExtent,
    ) -> TokenWindow:
        channel_ids = [channel_id for channel_id, _ in statics]
        values = [value for _, value in statics]
        times = [0.0] * len(statics)
        gaps = [0.0] * len(statics)
        timeless = [True] * len(statics)
        previous: dict[int, float] = {}
        repeated = False
        for time, channel_id, value in sorted(buffer):
            position = (time - extent.start) / extent.length
            last = previous.get(channel_id)
            repeated = repeated or last == position
            channel_ids.append(channel_id)
            values.append(value)
            times.append(position)
            # The first token of a channel measures its gap from the start of the window.
            gaps.append(position if last is None else position - last)
            timeless.append(False)
            previous[channel_id] = position
        if repeated:
            # A channel observed twice at one instant yields tokens equal in all but gap, and the
            # one with the smaller gap sorts first; the buffer, sorted before any gap was known,
            # put it second.
            order = sorted(
                range(len(channel_ids)),
                key=lambda i: canonical_key(
                    channel_ids[i], values[i], times[i], gaps[i], timeless[i]
                ),
            )
            channel_ids = [channel_ids[i] for i in order]
            values = [values[i] for i in order]
            times = [times[i] for i in order]
            gaps = [gaps[i] for i in order]
            timeless = [timeless[i] for i in order]
        return TokenWindow(
            tuple(channel_ids), tuple(values), tuple(times), tuple(gaps), tuple(timeless)
        )


class _RunningMoments:
    """Welford's running mean and sum of squared deviations, numerically stable in one pass."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def statistics(self) -> ChannelStatistics:
        return ChannelStatistics(self.count, self.mean, math.sqrt(self.m2 / self.count))


def _entries_by_channel(scheme: TokenisationScheme, corpus: str) -> dict[str, VocabularyEntry]:
    return {entry.channel: entry for entry in scheme.vocabulary.entries_of(corpus)}


def _entry(
    entries: Mapping[str, VocabularyEntry], corpus: str, channel: str, *, timeless: bool
) -> VocabularyEntry:
    entry = entries.get(channel)
    if entry is None:
        raise UnknownChannelError(f"corpus {corpus!r} has no channel {channel!r}")
    if entry.timeless != timeless:
        kind = "timeless" if entry.timeless else "timed"
        arrived = "a static feature" if timeless else "an observation"
        raise ChannelKindMismatchError(
            f"channel {channel!r} of corpus {corpus!r} is {kind}, but {arrived} arrived on it"
        )
    return entry
