from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidPairedReadingError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.outcome_level import OutcomeLevel

# The half-width of a 95 % interval in standard errors, as the registered spread states it.
_Z = 1.96


@dataclass(frozen=True, kw_only=True)
class PairedReading:
    """One paired difference as a rule reads it: its mean and error over seeds, its interval.

    The interval resamples the units and holds the seeds as they ran; the standard error over
    seeds sees what the interval cannot, how far another draw of labels moves the mean. Both
    enter the rules read off a reading.

    Invariants: the mean and standard error are finite, the error not negative.

    Attributes:
        mean: The difference averaged over the seeds, candidate less control in the gain's sign.
        standard_error: Standard error of that mean over the seeds.
        interval: Where the difference lies over resamples of the units.
    """

    mean: float
    standard_error: float
    interval: BootstrapInterval

    def __post_init__(self) -> None:
        if not isfinite(self.mean):
            raise InvalidPairedReadingError(f"mean must be finite, got {self.mean}")
        if not isfinite(self.standard_error) or self.standard_error < 0.0:
            raise InvalidPairedReadingError(
                f"standard error must be finite and not negative, got {self.standard_error}"
            )

    @property
    def noise(self) -> float:
        """The variance of the mean that is not a pretraining's: over seeds plus over units."""
        half = (self.interval.high - self.interval.low) / 2.0
        return self.standard_error**2 + (half / _Z) ** 2

    def exceeds(self, threshold: DetectionThreshold) -> bool:
        return self.mean > threshold.value

    def falls_below(self, threshold: DetectionThreshold) -> bool:
        """Whether the mean lies below the threshold's negative: a loss that is detected."""
        return self.mean < -threshold.value

    def level(self, threshold: DetectionThreshold, *, least_gain: float) -> OutcomeLevel:
        """The level of this reading taken as a gain, at ``threshold`` and ``least_gain``."""
        if self.interval.above_zero and self.exceeds(threshold):
            return (
                OutcomeLevel.LEAST_GAIN if self.mean >= least_gain else OutcomeLevel.REAL_AND_SMALL
            )
        if self.interval.high < 0.0 and self.falls_below(threshold):
            return OutcomeLevel.HARM
        return OutcomeLevel.NOISE
