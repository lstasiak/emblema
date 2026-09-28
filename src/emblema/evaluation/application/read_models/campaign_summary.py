from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class CampaignSummary:
    """A campaign as a list shows it: what it compares, on what, and how far its grid got.

    A read model rather than the aggregate: the design's axes spelled out as text a client can
    show, the grid's progress as two counts, and nothing of the results themselves.

    Attributes:
        campaign_id: Identity of the campaign.
        task: The task every candidate answers.
        purpose: What the campaign is for.
        tier: Hardware class its runs are measured on.
        finished: Whether its grid ran whole and it was closed.
        opened_at: When it was designed.
        completed_at: When it was closed; ``None`` while any cell is pending.
        control: The candidate the others are measured against.
        endpoint: The candidate the campaign's single claim is about.
        endpoint_budget: The budget that claim is made at, as text.
        candidates: Every competitor, in the order the campaign reports them.
        budgets: Every budget, as text, in reporting order.
        seeds: Every repeat of a cell.
        measure: What every run is read by.
        cells_recorded: How many cells of the grid have run.
        cells_planned: How many the grid holds.
    """

    campaign_id: CampaignId
    task: TaskId
    purpose: RunPurpose
    tier: ComputeTier
    finished: bool
    opened_at: UtcDateTime
    completed_at: UtcDateTime | None
    control: CandidateRef
    endpoint: CandidateRef
    endpoint_budget: str
    candidates: tuple[CandidateRef, ...]
    budgets: tuple[str, ...]
    seeds: tuple[int, ...]
    measure: ErrorMeasure
    cells_recorded: int
    cells_planned: int

    @classmethod
    def of(cls, campaign: CampaignOverview) -> Self:
        design = campaign.design
        return cls(
            campaign_id=campaign.campaign_id,
            task=campaign.task,
            purpose=campaign.purpose,
            tier=campaign.tier,
            finished=campaign.is_finished,
            opened_at=campaign.opened_at,
            completed_at=campaign.completed_at,
            control=design.control,
            endpoint=design.endpoint,
            endpoint_budget=design.endpoint_budget.text(),
            candidates=tuple(candidate.ref for candidate in design.candidates),
            budgets=tuple(budget.text() for budget in design.budgets),
            seeds=design.seeds,
            measure=design.measure,
            cells_recorded=campaign.cells_recorded,
            cells_planned=len(design.cells()),
        )
