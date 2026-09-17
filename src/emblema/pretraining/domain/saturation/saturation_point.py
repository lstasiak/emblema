from dataclasses import dataclass
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidSaturationCurveError


@dataclass(frozen=True, kw_only=True)
class SaturationPoint:
    """One run of the saturation measurement: a share of the corpus and what it ended at.

    Each loss comes with the loss of the trivial predictor on the same side — the channel mean,
    which in normalised units is zero, so its error is the mean square of the values. The two
    sides of a split are not equally hard: a validation side that drew the units with the widest
    excursions scores every predictor worse, and a run is read against what its side allows
    rather than against the other side's numbers.

    Invariants: the fraction lies in ``(0, 1]``; the losses are finite and not negative, being
    mean squared errors; the references are finite and positive; the steps are positive, since a
    run that took none measured nothing.

    Attributes:
        fraction: Share of the corpus's training units the run read.
        training_loss: Mean reconstruction error over the last epoch's hidden training tokens.
        validation_loss: The same error on the whole validation side, at the end of the run.
        training_reference: Error of the trivial predictor on the training side the run read.
        validation_reference: Error of the trivial predictor on the validation side.
        steps: Optimiser steps the run took, which is what makes two points comparable.
    """

    fraction: float
    training_loss: float
    validation_loss: float
    training_reference: float
    validation_reference: float
    steps: int

    def __post_init__(self) -> None:
        if not isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise InvalidSaturationCurveError(f"fraction must lie in (0, 1], got {self.fraction}")
        for label, value in (
            ("training_loss", self.training_loss),
            ("validation_loss", self.validation_loss),
        ):
            if not isfinite(value) or value < 0.0:
                raise InvalidSaturationCurveError(
                    f"{label} must be finite and not negative, got {value}"
                )
        for label, value in (
            ("training_reference", self.training_reference),
            ("validation_reference", self.validation_reference),
        ):
            if not isfinite(value) or value <= 0.0:
                raise InvalidSaturationCurveError(
                    f"{label} must be finite and positive, got {value}"
                )
        if self.steps < 1:
            raise InvalidSaturationCurveError(f"steps must be positive, got {self.steps}")

    @property
    def relative_validation_loss(self) -> float:
        """Validation loss over the trivial predictor's: under one, something was learnt."""
        return self.validation_loss / self.validation_reference

    @property
    def relative_training_loss(self) -> float:
        return self.training_loss / self.training_reference

    @property
    def generalisation_ratio(self) -> float:
        """How much worse the run does on windows it never saw, each side against its own floor.

        The ratio of the relative losses: a validation side that is harder for every predictor
        does not read as memorising.
        """
        if self.relative_training_loss <= 0.0:
            return float("inf")
        return self.relative_validation_loss / self.relative_training_loss
