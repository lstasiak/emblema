from enum import Enum


class Normalisation(Enum):
    """How a publication turns a channel's raw value into the value a model reads.

    Chosen once per publication for the channels it fits; the statistics of each channel record
    which one they serve, so a value is normalised the same way wherever the statistics travel.

    Attributes:
        STANDARD: The standard score, the distance from the mean in standard deviations.
        ROBUST: The distance from the median in robust deviations (1.4826 times the median
            absolute deviation), bounded, so a heavy tail cannot set the scale of the bulk.
        QUANTILE: The normal score of the value's rank among the training values, so every
            channel arrives with the same spread whatever the shape of its distribution.
    """

    STANDARD = "standard"
    ROBUST = "robust"
    QUANTILE = "quantile"

    def __str__(self) -> str:
        return self.value
