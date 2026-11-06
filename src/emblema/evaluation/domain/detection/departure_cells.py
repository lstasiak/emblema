from enum import StrEnum


class DepartureCells(StrEnum):
    """What of a window each channel's departure from its own history is measured on.

    Attributes:
        WINDOW: The channel's summary over the whole window: the mean of its spans' means, the
            least minimum, the largest maximum, the last reading and the count of readings.
        BINS: The same five values of every span of the window, each against the channel's
            spans of the windows before it, so a change within part of the window is not
            averaged over the rest.
    """

    WINDOW = "window"
    BINS = "bins"
