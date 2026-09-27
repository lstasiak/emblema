from collections.abc import Sequence
from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from emblema.evaluation.adapters.persistence.campaign_design_document import (
    CampaignDesignDocument,
)
from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.campaign_overview import CampaignOverview
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.compute import ComputeTier

RUNNING, FINISHED = "running", "finished"
DESIGNS = CampaignDesignDocument()


class EvaluationCampaignRecord(Base):
    """Row of ``evaluation.evaluation_campaign``: a campaign as it is stored, its cells apart.

    The task is referred to by identity alone and no foreign key stands behind it: a campaign
    and a task are separate aggregates, and a repository that saved one should not fail because
    of the order the other was written in. The design is a document, since it is settled once
    and never queried a field at a time, while the purpose, the tier and the status are columns
    of their own: what runs where, and how far it got, is asked of the whole table. The cells
    are rows of their own tables, keyed by the campaign and read by the repository as the
    campaign or its reading needs them, never through this row.
    """

    __tablename__ = "evaluation_campaign"
    __table_args__ = (
        CheckConstraint(f"status IN ('{RUNNING}', '{FINISHED}')", name="status_known"),
        CheckConstraint(
            f"(status = '{FINISHED}') = (completed_at IS NOT NULL)", name="finished_is_dated"
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    task_ref: Mapped[UUID] = mapped_column(Uuid)
    purpose: Mapped[str] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    # What the campaign's own revision counts, kept as a column so a writer can claim the
    # state it read in one statement instead of reading it again and hoping.
    version: Mapped[int] = mapped_column(Integer)
    design: Mapped[dict[str, object]] = mapped_column(JSONB)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @classmethod
    def from_campaign(cls, campaign: EvaluationCampaign) -> Self:
        return cls(
            id=campaign.campaign_id.value,
            task_ref=campaign.task.value,
            purpose=campaign.purpose.value,
            tier=str(campaign.tier),
            design=DESIGNS.encode(campaign.design),
            opened_at=campaign.opened_at.value,
            **cls.progress_of(campaign),
        )

    @staticmethod
    def progress_of(campaign: EvaluationCampaign) -> dict[str, object]:
        """The columns that move after a campaign is designed: how far it got, and its revision."""
        return {
            "status": FINISHED if campaign.is_finished else RUNNING,
            "version": campaign.revision,
            "completed_at": (
                None if campaign.completed_at is None else campaign.completed_at.value
            ),
        }

    def to_campaign(self, recorded: Sequence[CampaignCell]) -> EvaluationCampaign:
        """The campaign this row stores, ``recorded`` being the cells whose rows exist."""
        return EvaluationCampaign(
            campaign_id=CampaignId(self.id),
            task=TaskId(self.task_ref),
            purpose=RunPurpose(self.purpose),
            tier=ComputeTier(self.tier),
            design=DESIGNS.decode(dict(self.design)),
            recorded=tuple(recorded),
            opened_at=as_utc(self.opened_at),
            completed_at=None if self.completed_at is None else as_utc(self.completed_at),
        )

    def to_overview(self, cells_recorded: int) -> CampaignOverview:
        """The campaign without its cells, which a list counts rather than reads."""
        return CampaignOverview(
            campaign_id=CampaignId(self.id),
            task=TaskId(self.task_ref),
            purpose=RunPurpose(self.purpose),
            tier=ComputeTier(self.tier),
            design=DESIGNS.decode(dict(self.design)),
            opened_at=as_utc(self.opened_at),
            completed_at=None if self.completed_at is None else as_utc(self.completed_at),
            cells_recorded=cells_recorded,
        )
