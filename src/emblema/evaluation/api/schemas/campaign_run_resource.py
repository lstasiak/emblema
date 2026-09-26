from typing import Self

from pydantic import BaseModel, Field

from emblema.evaluation.application.read_models.campaign_run import CampaignRun


class CampaignRunResource(BaseModel):
    """One cell of a grid that has run."""

    candidate: str
    budget: str
    seed: int
    error: float = Field(description="Root mean squared error over the units scored.")
    units: int
    seconds: float
    artifact_key: str | None
    artifact_checksum: str | None

    @classmethod
    def of(cls, run: CampaignRun) -> Self:
        return cls(
            candidate=str(run.candidate),
            budget=run.budget,
            seed=run.seed,
            error=run.error,
            units=run.units,
            seconds=run.seconds,
            artifact_key=None if run.artifact is None else run.artifact.key,
            artifact_checksum=None if run.artifact is None else str(run.artifact.checksum),
        )
