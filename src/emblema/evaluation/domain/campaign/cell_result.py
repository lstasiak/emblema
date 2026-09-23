from dataclasses import dataclass
from math import isfinite, sqrt

from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.exceptions import InvalidCellResultError
from emblema.evaluation.domain.transfer.unit_error import UnitError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CellResult:
    """What one cell of a campaign produced, in the form the comparison consumes.

    The error is kept per unit rather than as one figure, because that is what a paired
    comparison resamples and what repeats of a cell are pooled over; the figure is derived from
    it here, so a result cannot disagree with itself. What the run cost is carried alongside,
    since a comparison of methods that ignores what each spent is not a comparison.

    Invariants: at least one unit scored, no unit scored twice; the time taken is finite and not
    negative.

    Attributes:
        cell: Which point of the grid this is.
        errors: What the candidate got wrong over each unit it was scored on.
        seconds: What the run took, learning and scoring together.
        artifact: The candidate as this run fitted it, where the campaign asked for it to be
            kept; ``None`` for every cell it did not.
    """

    cell: CampaignCell
    errors: tuple[UnitError, ...]
    seconds: float
    artifact: ArtifactRef | None

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

    @property
    def rmse(self) -> float:
        """The root mean squared error over every unit the run was scored on."""
        squared = sum(error.squared_error for error in self.errors)
        windows = sum(error.windows for error in self.errors)
        return sqrt(squared / windows)
