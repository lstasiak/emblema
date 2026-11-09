from enum import StrEnum

from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction


class ErrorNormalisation(StrEnum):
    """How the errors of a window's cells become its score, when a detector scores by an error.

    The errors come in the corpus's normalised units, yet a pretext misses some channels by far
    more than others however normal the window: a channel with heavy tails is missed widely
    everywhere, a nearly constant one is answered to a hair. Read raw, the mean over a window's
    cells follows the channels missed widely and never hears the quiet ones. Read per channel,
    each channel's error is divided by how much that channel is missed on the fitting side, so a
    quiet channel that departs counts as much as a loud one. The window then scores the mean
    over its channels, or the largest of them, which reads an event in a few channels of many as
    high as one in all of them, as the detector over the channels' departures reads it.

    Attributes:
        RAW: The mean error over the window's scored cells, in the corpus's normalised units.
        PER_CHANNEL: The mean over the window's channels of each channel's error over its
            typical error on the fitting side, that typical error floored so a channel the
            pretext answers exactly does not swamp the rest.
        LARGEST: The largest over the window's channels of the same errors over the same scales.
    """

    RAW = "raw"
    PER_CHANNEL = "channel"
    LARGEST = "largest"

    @property
    def scaled(self) -> bool:
        """Whether each channel's error is read over its own scale."""
        return self is not ErrorNormalisation.RAW

    @property
    def channels(self) -> ChannelReduction:
        """How the channels' errors over their scales become the window's score."""
        return ChannelReduction.MAX if self is ErrorNormalisation.LARGEST else ChannelReduction.MEAN
