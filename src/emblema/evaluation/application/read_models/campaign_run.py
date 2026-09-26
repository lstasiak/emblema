from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CampaignRun:
    """One cell of a grid that has run, as a list of runs shows it.

    The error per unit stays inside the aggregate, where the comparison resamples it; what a
    list shows is the figure, what the run cost, and what it kept.

    Attributes:
        candidate: Which competitor the cell measures.
        budget: How many labelled windows it learnt from, as text.
        seed: Which repeat of the pairing it is.
        error: The root mean squared error over every unit it was scored on.
        units: How many units it was scored on.
        seconds: What the run took, learning and scoring together.
        artifact: The fitted candidate, where the campaign kept it.
    """

    candidate: CandidateRef
    budget: str
    seed: int
    error: float
    units: int
    seconds: float
    artifact: ArtifactRef | None

    @classmethod
    def of(cls, result: CellResult) -> Self:
        return cls(
            candidate=result.cell.candidate,
            budget=result.cell.budget.text(),
            seed=result.cell.seed,
            error=result.rmse,
            units=len(result.errors),
            seconds=result.seconds,
            artifact=result.artifact,
        )
