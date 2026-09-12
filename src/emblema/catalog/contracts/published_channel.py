from dataclasses import dataclass

from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics


@dataclass(frozen=True, kw_only=True)
class PublishedChannel:
    """One entry of the vocabulary a corpus was tokenised under.

    Attributes:
        channel_id: Identifier the tokens of this channel carry; entry ``i`` of a manifest's
            channels carries ``i + 1``, so the list is the embedding table's index.
        corpus: Name of the corpus the channel belongs to.
        channel: Name of the channel within that corpus.
        timeless: Whether the channel is a static feature of the units rather than a measurement.
        unit: Physical unit of the values, when the source documents one.
        statistics: Normalisation of the channel's values; ``None`` for a channel the training
            units of this corpus never observed.
    """

    channel_id: int
    corpus: str
    channel: str
    timeless: bool = False
    unit: str | None = None
    statistics: PublishedChannelStatistics | None = None
