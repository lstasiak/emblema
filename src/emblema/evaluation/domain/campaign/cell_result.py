from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Self

from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.exceptions import InvalidCellResultError, PredictionsNotKeptError
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CellResult:
    """What one cell of a campaign produced, in the form the comparison consumes.

    The error is kept per unit rather than as one figure, because that is what a paired
    comparison resamples and what repeats of a cell are pooled over; the figure is derived from
    it here, so a result cannot disagree with itself. The answers themselves are kept beside it,
    because a measure read from how the answers rank does not split into sums per unit; a cell
    recorded before answers were kept holds none and is read by squared error alone. What the
    run cost is carried alongside, since a comparison of methods that ignores what each spent is
    not a comparison.

    Invariants: at least one unit scored, no unit scored twice; answers, where kept, cover the
    same units over the same number of windows as the errors; the time taken is finite and not
    negative.

    Attributes:
        cell: Which point of the grid this is.
        errors: What the candidate got wrong over each unit it was scored on.
        predictions: The candidate's answer for every window it was scored on; empty for a cell
            recorded before answers were kept.
        seconds: What the run took, learning and scoring together.
        artifact: The candidate as this run fitted it, where the campaign asked for it to be
            kept; ``None`` for every cell it did not.
    """

    cell: CampaignCell
    errors: tuple[UnitError, ...]
    seconds: float
    artifact: ArtifactRef | None
    predictions: tuple[WindowPrediction, ...] = ()

    def __post_init__(self) -> None:
        if not self.errors:
            raise InvalidCellResultError(f"the result of {self.cell} scores no unit")
        units = [error.unit for error in self.errors]
        if len(set(units)) != len(units):
            raise InvalidCellResultError(f"the result of {self.cell} scores a unit twice")
        if not isfinite(self.seconds) or self.seconds < 0.0:
            raise InvalidCellResultError(
                f"seconds must be finite and not negative, got {self.seconds}"
            )
        if self.predictions:
            answered = [(e.unit, e.windows) for e in UnitError.per_unit(self.predictions)]
            scored = sorted(((e.unit, e.windows) for e in self.errors), key=lambda e: str(e[0]))
            if answered != scored:
                raise InvalidCellResultError(
                    f"the answers of {self.cell} cover other units or windows than its errors"
                )

    @classmethod
    def of(cls, cell: CampaignCell, outcome: ScoredOutcome) -> Self:
        """What ``outcome`` says about ``cell``, whichever kind of candidate produced it."""
        return cls(
            cell=cell,
            errors=outcome.by_unit(),
            seconds=outcome.seconds,
            artifact=outcome.artifact,
            predictions=outcome.predictions,
        )

    @property
    def rmse(self) -> float:
        """The root mean squared error over every unit the run was scored on."""
        return sqrt(self.mean_squared_error)

    @property
    def mean_squared_error(self) -> float:
        """The mean squared error over every window; for probabilities, the Brier score."""
        squared = sum(error.squared_error for error in self.errors)
        windows = sum(error.windows for error in self.errors)
        return squared / windows

    def error_under(self, measure: ErrorMeasure) -> float:
        """What the run scored under ``measure``, as an error.

        Raises:
            PredictionsNotKeptError: If the measure reads the answers and the cell kept none.
            InvalidWindowRankingError: If the answers hold one outcome only.
        """
        match measure:
            case ErrorMeasure.RMSE:
                return self.rmse
            case ErrorMeasure.AUROC_SHORTFALL:
                return 1.0 - self.ranking().auroc

    def ranking(self) -> WindowRanking:
        """The run's answers in order, for a measure read from how they rank.

        Raises:
            PredictionsNotKeptError: If the cell kept no answers.
            InvalidWindowRankingError: If the answers hold one outcome only.
        """
        if not self.predictions:
            raise PredictionsNotKeptError(f"{self.cell} was recorded without its answers")
        return WindowRanking.of(self.predictions)
