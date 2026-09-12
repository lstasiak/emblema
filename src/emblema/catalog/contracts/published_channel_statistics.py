from dataclasses import dataclass


@dataclass(frozen=True)
class PublishedChannelStatistics:
    """Parameters the training units established for one channel's normalisation.

    Attributes:
        count: Values the statistics were computed over.
        mean: Arithmetic mean.
        std: Population standard deviation.
    """

    count: int
    mean: float
    std: float
