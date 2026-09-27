from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, log1p
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import (
    InvalidLogisticCalibrationError,
    UncalibratableScoresError,
)
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme


@dataclass(frozen=True, kw_only=True)
class LogisticCalibration:
    """Platt scaling: a sigmoid of an affine function of a score, giving a probability.

    A least-squares head over outcomes ranks them but answers no probabilities. The map is fitted
    by maximum likelihood on scores the head did not see while it was fitted, and it is monotone,
    so the ranking and the area under the ROC curve are unchanged. The targets are Platt's
    smoothed ones, ``(n+ + 1) / (n+ + 2)`` and ``1 / (n- + 2)``, which keep the maximum finite
    when the scores separate the outcomes. The optimiser is Newton's method with a backtracking
    line search (Lin, Lin and Weng, 2007).

    Invariants: the slope and the intercept are finite.

    Attributes:
        slope: How much a unit of score moves the log-odds.
        intercept: The log-odds at a score of zero.
    """

    slope: float
    intercept: float

    # Lin, Lin and Weng's settings: a step is accepted when it buys at least this share of the
    # decrease its gradient promises, halved down to the smallest step; the fit stops once no
    # gradient component exceeds the tolerance; the ridge keeps the Hessian invertible when
    # every score is the same.
    _ARMIJO: ClassVar[float] = 1e-4
    _SMALLEST_STEP: ClassVar[float] = 1e-10
    _TOLERANCE: ClassVar[float] = 1e-5
    _ITERATIONS: ClassVar[int] = 100
    _RIDGE: ClassVar[float] = 1e-12

    def __post_init__(self) -> None:
        if not (isfinite(self.slope) and isfinite(self.intercept)):
            raise InvalidLogisticCalibrationError(
                f"a calibration must be finite, got {self.slope} and {self.intercept}"
            )

    @classmethod
    def identity(cls) -> Self:
        """The calibration that leaves a log-odds as it is: slope one, intercept zero."""
        return cls(slope=1.0, intercept=0.0)

    @classmethod
    def fitted(cls, scores: Sequence[float], outcomes: Sequence[float]) -> Self:
        """The calibration of ``scores`` against ``outcomes`` by Platt's method.

        Raises:
            UncalibratableScoresError: If the scores and outcomes differ in number, an outcome
                is not zero or one, the outcomes are all one kind, a score is not finite, or
                the fit does not converge.
        """
        if len(scores) != len(outcomes):
            raise UncalibratableScoresError(
                f"{len(scores)} scores cannot be calibrated against {len(outcomes)} outcomes"
            )
        if any(outcome not in OutcomeScheme.OUTCOMES for outcome in outcomes):
            raise UncalibratableScoresError("every outcome must be zero or one")
        if not all(isfinite(score) for score in scores):
            raise UncalibratableScoresError("every score must be finite")
        positives = sum(1 for outcome in outcomes if outcome == 1.0)
        negatives = len(outcomes) - positives
        if positives == 0 or negatives == 0:
            raise UncalibratableScoresError("a calibration needs both outcomes")
        high = (positives + 1.0) / (positives + 2.0)
        low = 1.0 / (negatives + 2.0)
        targets = [high if outcome == 1.0 else low for outcome in outcomes]
        slope, intercept = 0.0, log((positives + 1.0) / (negatives + 1.0))
        loss = cls._loss(scores, targets, slope, intercept)
        for _ in range(cls._ITERATIONS):
            gradient_slope = gradient_intercept = 0.0
            curvature_ss = curvature_si = curvature_ii = cls._RIDGE
            for score, target in zip(scores, targets, strict=True):
                probability = cls._sigmoid(slope * score + intercept)
                residual = probability - target
                gradient_slope += residual * score
                gradient_intercept += residual
                weight = probability * (1.0 - probability)
                curvature_ss += weight * score * score
                curvature_si += weight * score
                curvature_ii += weight
            if max(abs(gradient_slope), abs(gradient_intercept)) < cls._TOLERANCE:
                return cls(slope=slope, intercept=intercept)
            determinant = curvature_ss * curvature_ii - curvature_si * curvature_si
            step_slope = -(curvature_ii * gradient_slope - curvature_si * gradient_intercept)
            step_slope /= determinant
            step_intercept = -(curvature_ss * gradient_intercept - curvature_si * gradient_slope)
            step_intercept /= determinant
            promised = gradient_slope * step_slope + gradient_intercept * step_intercept
            step = 1.0
            while step >= cls._SMALLEST_STEP:
                trial_slope = slope + step * step_slope
                trial_intercept = intercept + step * step_intercept
                trial = cls._loss(scores, targets, trial_slope, trial_intercept)
                if trial < loss + cls._ARMIJO * step * promised:
                    slope, intercept, loss = trial_slope, trial_intercept, trial
                    break
                step /= 2.0
            else:
                raise UncalibratableScoresError(
                    "the line search found no step that lowers the loss"
                )
        raise UncalibratableScoresError(f"the fit did not converge in {cls._ITERATIONS} iterations")

    def log_odds(self, score: float) -> float:
        """The log-odds of the positive outcome at ``score``."""
        return self.slope * score + self.intercept

    def probability(self, score: float) -> float:
        """The probability of the positive outcome at ``score``."""
        return self._sigmoid(self.log_odds(score))

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0.0:
            return 1.0 / (1.0 + exp(-z))
        shrunk = exp(z)
        return shrunk / (1.0 + shrunk)

    @staticmethod
    def _loss(
        scores: Sequence[float], targets: Sequence[float], slope: float, intercept: float
    ) -> float:
        """The cross-entropy of the smoothed targets, computed without overflow at any log-odds."""
        total = 0.0
        for score, target in zip(scores, targets, strict=True):
            z = slope * score + intercept
            softplus = z + log1p(exp(-z)) if z >= 0.0 else log1p(exp(z))
            total += softplus - target * z
        return total
