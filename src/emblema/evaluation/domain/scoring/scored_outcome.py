from dataclasses import dataclass
from math import isfinite, sqrt

from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class ScoredOutcome:
    """What one run of any candidate produced: its answer per window, and what it cost.

    The part every kind of candidate shares, whatever it is made of. A fitted set of trees has
    no epochs to report a loss for and no weights to count, so an outcome that carried those
    would carry zeros for it and invite a reader to compare them with a network's; what is here
    is what a comparison actually consumes. A kind with more to say about its run extends this
    rather than repeating it, so the rules an answer is held to are stated once.

    Invariants: at least one prediction, no window predicted twice; the time taken is finite and
    not negative.

    Attributes:
        predictions: The candidate's answer for every scored window, in the order given.
        seconds: What the run took, learning and scoring together.
        artifact: The candidate as this run left it, where it was asked to keep it; ``None``
            otherwise. A reference, never the model itself.
    """

    predictions: tuple[WindowPrediction, ...]
    seconds: float
    artifact: ArtifactRef | None

    def __post_init__(self) -> None:
        if not self.predictions:
            raise InvalidScoredOutcomeError("a run must predict at least one window")
        places = [(str(p.window.unit), p.window.position) for p in self.predictions]
        if len(set(places)) != len(places):
            raise InvalidScoredOutcomeError("a window is predicted twice")
        if not isfinite(self.seconds) or self.seconds < 0.0:
            raise InvalidScoredOutcomeError(
                f"seconds must be finite and not negative, got {self.seconds}"
            )

    @property
    def rmse(self) -> float:
        """The root mean squared error over every scored window."""
        return sqrt(sum(p.squared_error for p in self.predictions) / len(self.predictions))

    def by_unit(self) -> tuple[UnitError, ...]:
        """The error per scored unit, in unit order — what a paired comparison resamples."""
        return UnitError.per_unit(self.predictions)
