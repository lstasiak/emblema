from enum import StrEnum


class ErrorNormalisation(StrEnum):
    """How the errors of a window's cells become its score, when a detector scores by an error.

    The errors come in the corpus's normalised units, yet a pretext misses some channels by far
    more than others however normal the window: a channel with heavy tails is missed widely
    everywhere, a nearly constant one is answered to a hair. Read raw, the mean over a window's
    cells follows the channels missed widely and never hears the quiet ones. Read per channel,
    each channel's error is divided by how much that channel is missed on the fitting side, so a
    quiet channel that departs counts as much as a loud one.

    Attributes:
        RAW: The mean error over the window's scored cells, in the corpus's normalised units.
        PER_CHANNEL: The mean over the window's channels of each channel's error over its
            typical error on the fitting side, that typical error floored so a channel the
            pretext answers exactly does not swamp the rest.
    """

    RAW = "raw"
    PER_CHANNEL = "channel"
