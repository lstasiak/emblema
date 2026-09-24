from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.evaluation.adapters.persistence.campaign_cell_record import CampaignCellRecord
from emblema.evaluation.adapters.persistence.campaign_design_document import (
    CampaignDesignDocument,
)
from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.compute import ComputeTier

RUNNING, FINISHED = "running", "finished"
DESIGNS = CampaignDesignDocument()


class EvaluationCampaignRecord(Base):
    """Row of ``evaluation.evaluation_campaign`` with its cells: a campaign as it is stored.

    The task is referred to by identity alone and no foreign key stands behind it: a campaign
    and a task are separate aggregates, and a repository that saved one should not fail because
    of the order the other was written in. The design is a document, since it is settled once
    and never queried a field at a time, while the purpose, the tier and the status are columns
    of their own: what runs where, and how far it got, is asked of the whole table.
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
    cells: Mapped[list[CampaignCellRecord]] = relationship(
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=(
            CampaignCellRecord.candidate,
            CampaignCellRecord.budget,
            CampaignCellRecord.seed,
        ),
    )

    @classmethod
    def from_campaign(cls, campaign: EvaluationCampaign) -> Self:
        return cls(
            id=campaign.campaign_id.value,
            task_ref=campaign.task.value,
            purpose=campaign.purpose.value,
            tier=str(campaign.tier),
            status=FINISHED if campaign.is_finished else RUNNING,
            version=campaign.revision,
            design=DESIGNS.encode(campaign.design),
            opened_at=campaign.opened_at.value,
            completed_at=(None if campaign.completed_at is None else campaign.completed_at.value),
            cells=[
                CampaignCellRecord.from_result(campaign.campaign_id, result)
                for result in campaign.results
            ],
        )

    def to_campaign(self) -> EvaluationCampaign:
        return EvaluationCampaign(
            campaign_id=CampaignId(self.id),
            task=TaskId(self.task_ref),
            purpose=RunPurpose(self.purpose),
            tier=ComputeTier(self.tier),
            design=DESIGNS.decode(dict(self.design)),
            results=tuple(record.to_result() for record in self.cells),
            opened_at=as_utc(self.opened_at),
            completed_at=None if self.completed_at is None else as_utc(self.completed_at),
        )
