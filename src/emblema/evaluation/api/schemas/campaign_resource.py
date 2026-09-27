from datetime import datetime
from typing import Self

from pydantic import BaseModel

from emblema.evaluation.application.read_models.campaign_summary import CampaignSummary


class CampaignResource(BaseModel):
    """A campaign: what it compares, on what, and how far its grid got."""

    campaign_id: str
    task_id: str
    purpose: str
    tier: str
    finished: bool
    opened_at: datetime
    completed_at: datetime | None
    control: str
    endpoint: str
    endpoint_budget: str
    candidates: list[str]
    budgets: list[str]
    seeds: list[int]
    cells_recorded: int
    cells_planned: int

    @classmethod
    def of(cls, summary: CampaignSummary) -> Self:
        return cls(
            campaign_id=str(summary.campaign_id),
            task_id=str(summary.task),
            purpose=summary.purpose.value,
            tier=str(summary.tier),
            finished=summary.finished,
            opened_at=summary.opened_at.value,
            completed_at=None if summary.completed_at is None else summary.completed_at.value,
            control=str(summary.control),
            endpoint=str(summary.endpoint),
            endpoint_budget=summary.endpoint_budget,
            candidates=[str(candidate) for candidate in summary.candidates],
            budgets=list(summary.budgets),
            seeds=list(summary.seeds),
            cells_recorded=summary.cells_recorded,
            cells_planned=summary.cells_planned,
        )
