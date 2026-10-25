from enum import StrEnum


class OutcomeLevel(StrEnum):
    """Where a gain of fine-tuning over the network from nothing lies, as the register reads it.

    A detected gain is not a success: the level says how large it is beside the least gain the
    claim asks for, and harm is a level of its own rather than the absence of a gain.

    Attributes:
        LEAST_GAIN: The interval lies above zero, the gain exceeds the threshold and reaches the
            least gain.
        REAL_AND_SMALL: The interval lies above zero and the gain exceeds the threshold, below
            the least gain.
        HARM: The interval lies below zero and the loss exceeds the threshold.
        NOISE: Every other case.
    """

    LEAST_GAIN = "at the least gain"
    REAL_AND_SMALL = "real and small"
    HARM = "harm"
    NOISE = "noise"
