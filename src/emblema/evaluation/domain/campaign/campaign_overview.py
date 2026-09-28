from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign
from emblema.evaluation.domain.campaign.campaign_position import CampaignPosition
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import InvalidCampaignOverviewError
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True, kw_only=True)
class CampaignOverview:
    """A campaign without its results: the design, and how many cells of its grid have run.

    What a list of campaigns is read as. A campaign's results are the bulk of it — every cell
    with every unit's error — and a list shows none of them, only how far the grid got, so a
    registry answers a list with this rather than rebuilding each campaign whole.

    Invariants: the count of cells run lies within the grid, and a finished campaign ran all.

    Attributes:
        campaign_id: Identity of the campaign.
        task: Task every candidate answers.
        purpose: What the campaign is for.
        tier: Hardware class its runs are measured on.
        design: Who competes, over what, and how the results will be read.
        opened_at: When the campaign was designed.
        completed_at: When its grid was finished; ``None`` while any cell is pending.
        cells_recorded: How many cells of the grid have run.
    """

    campaign_id: CampaignId
    task: TaskId
    purpose: RunPurpose
    tier: ComputeTier
    design: CampaignDesign
    opened_at: UtcDateTime
    completed_at: UtcDateTime | None
    cells_recorded: int

    def __post_init__(self) -> None:
        planned = len(self.design.cells())
        if not 0 <= self.cells_recorded <= planned:
            raise InvalidCampaignOverviewError(
                f"{self.cells_recorded} cells recorded of a grid of {planned}"
            )
        if self.completed_at is not None and self.cells_recorded != planned:
            raise InvalidCampaignOverviewError(
                f"a campaign finished with {planned - self.cells_recorded} cells still pending"
            )

    @classmethod
    def of(cls, campaign: EvaluationCampaign) -> Self:
        return cls(
            campaign_id=campaign.campaign_id,
            task=campaign.task,
            purpose=campaign.purpose,
            tier=campaign.tier,
            design=campaign.design,
            opened_at=campaign.opened_at,
            completed_at=campaign.completed_at,
            cells_recorded=campaign.cells_recorded,
        )

    @property
    def is_finished(self) -> bool:
        return self.completed_at is not None

    @property
    def position(self) -> CampaignPosition:
        return CampaignPosition(opened_at=self.opened_at, campaign_id=self.campaign_id)
