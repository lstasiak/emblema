from dataclasses import dataclass
from math import isfinite, sqrt

from emblema.evaluation.domain.exceptions import InvalidClassicalOutcomeError
from emblema.evaluation.domain.transfer.unit_error import UnitError
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class ClassicalOutcome:
    """What fitting one classical candidate produced: its answer per window, and what it cost.

    Poorer than an adaptation's outcome, and deliberately: a fitted set of trees has no epochs
    to report a loss for and no weights to count, so the fields that would carry those would
    carry zeros and invite a reader to compare them with a network's. What is here is what a
    comparison actually consumes — an answer per scored window, the time the fit took, and the
    artifact where the candidate was kept if the campaign asked for it.

    Invariants: at least one prediction, no window predicted twice; the time taken is finite and
    not negative.

    Attributes:
        predictions: The candidate's answer for every scored window, in the order given.
        seconds: What the fit took, learning and scoring together.
        artifact: The candidate as this fit left it, where it was asked to keep it; ``None``
            otherwise. A reference, never the model itself.
    """

    predictions: tuple[WindowPrediction, ...]
    seconds: float
    artifact: ArtifactRef | None

    def __post_init__(self) -> None:
        if not self.predictions:
            raise InvalidClassicalOutcomeError("a fit must predict at least one window")
        places = [(str(p.window.unit), p.window.position) for p in self.predictions]
        if len(set(places)) != len(places):
            raise InvalidClassicalOutcomeError("a window is predicted twice")
        if not isfinite(self.seconds) or self.seconds < 0.0:
            raise InvalidClassicalOutcomeError(
                f"seconds must be finite and not negative, got {self.seconds}"
            )

    @property
    def rmse(self) -> float:
        """The root mean squared error over every scored window."""
        return sqrt(sum(p.squared_error for p in self.predictions) / len(self.predictions))

    def by_unit(self) -> tuple[UnitError, ...]:
        """The error per scored unit, in unit order — what a paired comparison resamples."""
        return UnitError.per_unit(self.predictions)
