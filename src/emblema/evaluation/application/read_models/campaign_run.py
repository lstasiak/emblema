from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
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
        measure: What the campaign reads its runs by.
        error: The run's error under that measure, over every unit it was scored on.
        brier: The mean squared error of its probabilities, where the answers are probabilities;
            ``None`` otherwise.
        units: How many units it was scored on.
        seconds: What the run took, learning and scoring together.
        artifact: The fitted candidate, where the campaign kept it.
    """

    candidate: CandidateRef
    budget: str
    seed: int
    measure: ErrorMeasure
    error: float
    brier: float | None
    units: int
    seconds: float
    artifact: ArtifactRef | None

    @classmethod
    def of(cls, result: CellResult, measure: ErrorMeasure) -> Self:
        return cls(
            candidate=result.cell.candidate,
            budget=result.cell.budget.text(),
            seed=result.cell.seed,
            measure=measure,
            error=result.error_under(measure),
            brier=result.mean_squared_error if measure is ErrorMeasure.AUROC_SHORTFALL else None,
            units=len(result.errors),
            seconds=result.seconds,
            artifact=result.artifact,
        )
