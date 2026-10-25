from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidDetectionThresholdError


@dataclass(frozen=True, kw_only=True)
class DetectionThreshold:
    """How far a mean over seeds must lie from zero before a comparison counts as detected.

    Twice the root of the mean's own variance over the draws of labels and of the variance a
    pretraining adds. A side read under n pretrainings adds s²/2 over n, s being the spread a
    difference between two pretrainings takes from pretraining alone, so half of it is one
    pretraining's share; a side without pretraining, the network from nothing, adds nothing.

    Invariants: the value is finite and not negative.

    Attributes:
        value: The threshold, in the difference's unit.
    """

    value: float

    def __post_init__(self) -> None:
        if not isfinite(self.value) or self.value < 0.0:
            raise InvalidDetectionThresholdError(
                f"a threshold must be finite and not negative, got {self.value}"
            )

    @classmethod
    def of_gain(cls, *, standard_error: float, spread: float, pretrainings: int) -> Self:
        """The threshold of fine-tuning under ``pretrainings`` against a side without any.

        Raises:
            InvalidDetectionThresholdError: If no pretraining is read, or a figure is negative
                or not finite.
        """
        return cls.of_contrast(standard_error=standard_error, spread=spread, sides=(pretrainings,))

    @classmethod
    def of_contrast(cls, *, standard_error: float, spread: float, sides: tuple[int, ...]) -> Self:
        """The threshold of a difference whose every side in ``sides`` was pretrained that often.

        Raises:
            InvalidDetectionThresholdError: If a side names no pretraining, or a figure is
                negative or not finite.
        """
        if not sides or any(count < 1 for count in sides):
            raise InvalidDetectionThresholdError(
                f"every pretrained side needs a pretraining, got {sides}"
            )
        for name, figure in (("standard error", standard_error), ("spread", spread)):
            if not isfinite(figure) or figure < 0.0:
                raise InvalidDetectionThresholdError(
                    f"the {name} must be finite and not negative, got {figure}"
                )
        share = sum(1.0 / count for count in sides)
        return cls(value=2.0 * sqrt(standard_error**2 + (spread**2 / 2.0) * share))
