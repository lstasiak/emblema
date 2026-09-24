from enum import StrEnum


class FeatureScheme(StrEnum):
    """What a classical candidate turns a window of tokens into before it fits anything.

    A window is a set of observations, not a table: the channels present differ from window to
    window and the instants are not on a grid. Every classical method needs a vector of fixed
    width, so the scheme is what bridges the two — and the choice of bridge decides what the
    candidate can be fitted over, which is why it is stated here rather than buried in an
    adapter. A vector laid out channel by channel is as wide as the corpus has channels and
    means nothing beside a vector from a corpus with other ones; a vector of statistics
    summarised across the channels has the same width whatever the layout, which is what makes
    a classical method capable of transfer at all.

    Attributes:
        PER_CHANNEL: One block of statistics per channel of the corpus, in vocabulary order.
            The strongest reading of a single corpus and the one that cannot leave it.
        CHANNEL_AGGREGATED: The same statistics summarised across whichever channels the window
            holds. Loses which channel said what, and in exchange fits over corpora whose
            channel layouts have nothing in common.
    """

    PER_CHANNEL = "per_channel"
    CHANNEL_AGGREGATED = "channel_aggregated"

    @property
    def spans_channel_layouts(self) -> bool:
        """Whether a fit under this scheme may take in windows from corpora of other layouts."""
        return self is FeatureScheme.CHANNEL_AGGREGATED
