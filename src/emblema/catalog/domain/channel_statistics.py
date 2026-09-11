import math
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidChannelStatisticsError


@dataclass(frozen=True)
class ChannelStatistics:
    """What the training data established about one channel, and the normalisation it implies.

    The statistics are facts about the data a scheme was fitted on — how many values, their mean
    and their population standard deviation — and are kept as such rather than folded into a
    transform, because they are also the provenance of the model and the only description of a
    channel the data itself offers. Normalisation is the z-score. A channel that never varied in
    training keeps its deviations in raw units instead of dividing by a vanishing spread: a value
    that does deviate later then stays finite and honest about how far it strayed.

    Attributes:
        count: Values the statistics were computed over; at least one.
        mean: Arithmetic mean; finite.
        std: Population standard deviation; finite, non-negative.
    """

    count: int
    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.count < 1:
            raise InvalidChannelStatisticsError(f"count must be positive, got {self.count}")
        if not math.isfinite(self.mean):
            raise InvalidChannelStatisticsError(f"mean must be finite, got {self.mean}")
        if not (math.isfinite(self.std) and self.std >= 0):
            raise InvalidChannelStatisticsError(
                f"std must be finite and non-negative, got {self.std}"
            )

    def normalise(self, value: float) -> float:
        return (value - self.mean) / (self.std if self.std > 0.0 else 1.0)
