from dataclasses import dataclass, replace
from typing import Self

from emblema.catalog.domain.channels.channel_schema import ChannelSchema
from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import (
    ChannelAlreadyFittedError,
    InvalidTokenisationSchemeError,
    MissingChannelStatisticsError,
)
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.window_reconstruction import WindowReconstruction
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class TokenisationScheme:
    """Everything a tokeniser needs to turn raw observations into tokens the same way every time.

    A scheme is the vocabulary plus the statistics of each channel, fitted on training units only.
    It travels with every tokenised corpus and every model, so that inference normalises a value
    exactly as training did. Statistics are fitted once per channel and never overwritten: a channel
    the training data did not observe has none, and tokenising it fails rather than passing a raw
    value through.

    Attributes:
        vocabulary: Channels the scheme knows.
        statistics: Statistics per vocabulary entry, aligned with it; ``None`` until fitted.
    """

    vocabulary: ChannelVocabulary
    statistics: tuple[ChannelStatistics | None, ...]

    def __post_init__(self) -> None:
        if len(self.statistics) != len(self.vocabulary):
            raise InvalidTokenisationSchemeError(
                f"statistics must align with the vocabulary: {len(self.statistics)} entries for "
                f"{len(self.vocabulary)} channels"
            )

    @classmethod
    def for_vocabulary(cls, vocabulary: ChannelVocabulary) -> Self:
        """A scheme over ``vocabulary`` with no channel fitted yet."""
        return cls(vocabulary, (None,) * len(vocabulary))

    def extended_with(self, corpus: str, schema: ChannelSchema) -> Self:
        """The scheme after registering the channels of ``schema`` under ``corpus``, unfitted.

        Raises:
            ChannelRedeclaredError: If a registered channel is declared differently.
        """
        vocabulary = self.vocabulary.extended_with(corpus, schema)
        added = len(vocabulary) - len(self.vocabulary)
        return replace(self, vocabulary=vocabulary, statistics=(*self.statistics, *(None,) * added))

    def with_statistics(self, channel_id: int, statistics: ChannelStatistics) -> Self:
        """The scheme with ``statistics`` recorded for a channel that had none.

        Raises:
            UnknownChannelError: If no channel carries the identifier.
            ChannelAlreadyFittedError: If the channel already has statistics.
        """
        entry = self.vocabulary.entry(channel_id)
        if self.statistics[channel_id - 1] is not None:
            raise ChannelAlreadyFittedError(
                f"channel {entry.channel!r} of {entry.corpus!r} is fitted"
            )
        updated = list(self.statistics)
        updated[channel_id - 1] = statistics
        return replace(self, statistics=tuple(updated))

    def statistics_of(self, channel_id: int) -> ChannelStatistics:
        """The statistics fitted for a channel.

        Raises:
            UnknownChannelError: If no channel carries the identifier.
            MissingChannelStatisticsError: If the channel was never observed in training data.
        """
        entry = self.vocabulary.entry(channel_id)
        statistics = self.statistics[channel_id - 1]
        if statistics is None:
            raise MissingChannelStatisticsError(
                f"channel {entry.channel!r} of {entry.corpus!r} has no statistics: it was not "
                f"observed in the training data"
            )
        return statistics

    def reconstruct(self, window: TokenWindow, extent: TimeExtent) -> WindowReconstruction:
        """The data ``window`` was cut from, in raw units, on the time axis ``extent`` spans.

        Tokenising one window run backwards: each token's channel by name, its value put back
        through the statistics that normalised it, and, for a timed token, the instant its
        position in the window stands for. Time is recovered from positions and never from gaps —
        the first token of a channel measures its gap from the start of the window, so it says
        nothing about when that channel was last observed.

        Args:
            window: Tokens to read back.
            extent: Span of the unit's time axis the window covers; the same one it was cut over.

        Raises:
            UnknownChannelError: If a token carries an identifier the vocabulary has not registered.
            MissingChannelStatisticsError: If a token's channel was never fitted.
        """
        observations: list[Observation] = []
        static_features: list[StaticFeature] = []
        for token in window:
            entry = self.vocabulary.entry(token.channel_id)
            value = self.statistics_of(token.channel_id).denormalise(token.value)
            if token.timeless:
                static_features.append(StaticFeature(entry.channel, value))
            else:
                observations.append(
                    Observation(entry.channel, extent.start + token.time * extent.length, value)
                )
        return WindowReconstruction(tuple(observations), tuple(static_features))
