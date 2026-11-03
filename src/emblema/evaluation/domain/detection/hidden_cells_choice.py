from enum import StrEnum


class HiddenCellsChoice(StrEnum):
    """Which of a window's cells a detector scored by an error asks the pretext to fill in.

    Under the backbone's own masks the hidden cells lie anywhere a training draw would put them,
    whole channels and blocks among them, and what is scored is how well the window is
    reconstructed from the rest of itself. Hiding the last span of every channel instead asks
    for the window's end from everything before it: a forecast, the reading published detectors
    of telemetry score by, where a departure in the latest readings is what an operator wants
    flagged. One such mask has nothing to draw, so it is asked once.

    Attributes:
        OWN: The backbone's own masks, drawn as in its training.
        TAIL: The last span of every channel, everything before it visible.
    """

    OWN = "own"
    TAIL = "tail"
