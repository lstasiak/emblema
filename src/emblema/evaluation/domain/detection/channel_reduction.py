from enum import StrEnum


class ChannelReduction(StrEnum):
    """How the departures of a window's channels become the window's one score.

    Attributes:
        MAX: The largest departure of any channel, so an event in a few channels of many
            scores as high as one in all of them.
        MEAN: The mean departure over the channels holding a reading in the window.
    """

    MAX = "max"
    MEAN = "mean"
