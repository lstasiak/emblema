from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field

from emblema.serving.application.read_models.served_model_summary import ServedModelSummary


class ServedModelResource(BaseModel):
    """A served model: identity, state, provenance and the artifact it serves."""

    served_model_id: str
    state: str
    campaign_id: str = Field(description="The finished campaign that measured the artifact.")
    task_id: str = Field(description="The task that campaign answered.")
    candidate: str = Field(description="What the campaign called the competitor.")
    kind: str
    artifact_key: str
    artifact_checksum: str
    promoted_at: datetime
    withdrawn_at: datetime | None

    @classmethod
    def of(cls, summary: ServedModelSummary) -> Self:
        return cls(
            served_model_id=str(summary.served_model_id),
            state=summary.state.value,
            campaign_id=str(summary.campaign),
            task_id=str(summary.task),
            candidate=str(summary.candidate),
            kind=summary.kind.value,
            artifact_key=summary.artifact.key,
            artifact_checksum=str(summary.artifact.checksum),
            promoted_at=summary.promoted_at.value,
            withdrawn_at=None if summary.withdrawn_at is None else summary.withdrawn_at.value,
        )
