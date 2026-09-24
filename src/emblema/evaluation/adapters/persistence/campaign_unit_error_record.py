from typing import Self
from uuid import UUID

from sqlalchemy import Float, ForeignKeyConstraint, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError


class CampaignUnitErrorRecord(Base):
    """Row of ``evaluation.campaign_unit_error``: what one cell got wrong over one unit.

    The long table the comparison is made of. It is kept per unit rather than as a summary
    because the unit is what a paired interval resamples, and a summary would leave every
    interval in the project unrecomputable from what was stored.
    """

    __tablename__ = "campaign_unit_error"
    __table_args__ = (
        ForeignKeyConstraint(
            ["campaign_id", "candidate", "budget", "seed"],
            [
                "campaign_cell.campaign_id",
                "campaign_cell.candidate",
                "campaign_cell.budget",
                "campaign_cell.seed",
            ],
            name="fk_campaign_unit_error_campaign_id",
            ondelete="CASCADE",
        ),
    )

    campaign_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    candidate: Mapped[str] = mapped_column(Text, primary_key=True)
    budget: Mapped[str] = mapped_column(Text, primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, primary_key=True)
    unit: Mapped[str] = mapped_column(Text, primary_key=True)
    squared_error: Mapped[float] = mapped_column(Float)
    windows: Mapped[int] = mapped_column(Integer)

    @classmethod
    def of(cls, campaign_id: CampaignId, cell: CampaignCell, error: UnitError) -> Self:
        return cls(
            campaign_id=campaign_id.value,
            candidate=str(cell.candidate),
            budget=cell.budget.text(),
            seed=cell.seed,
            unit=str(error.unit),
            squared_error=error.squared_error,
            windows=error.windows,
        )

    def to_error(self) -> UnitError:
        return UnitError(
            unit=UnitKey(self.unit), squared_error=self.squared_error, windows=self.windows
        )
