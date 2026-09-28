from typing import Self
from uuid import UUID

from sqlalchemy import Float, ForeignKeyConstraint, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction


class CampaignWindowPredictionRecord(Base):
    """Row of ``evaluation.campaign_window_prediction``: one cell's answer for one window.

    Kept beside the errors per unit because a measure read from how the answers rank does not
    split into sums per unit: the area under the ROC curve of a resample of the units, or of
    another question asked of the same run, can only be computed from the answers themselves.
    A cell recorded before answers were kept has no rows here.
    """

    __tablename__ = "campaign_window_prediction"
    __table_args__ = (
        ForeignKeyConstraint(
            ["campaign_id", "candidate", "budget", "seed"],
            [
                "campaign_cell.campaign_id",
                "campaign_cell.candidate",
                "campaign_cell.budget",
                "campaign_cell.seed",
            ],
            name="fk_campaign_window_prediction_campaign_id",
            ondelete="CASCADE",
        ),
    )

    campaign_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    candidate: Mapped[str] = mapped_column(Text, primary_key=True)
    budget: Mapped[str] = mapped_column(Text, primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, primary_key=True)
    unit: Mapped[str] = mapped_column(Text, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    ends_at: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    predicted: Mapped[float] = mapped_column(Float)

    @classmethod
    def of(cls, campaign_id: CampaignId, cell: CampaignCell, prediction: WindowPrediction) -> Self:
        return cls(
            campaign_id=campaign_id.value,
            candidate=str(cell.candidate),
            budget=cell.budget.text(),
            seed=cell.seed,
            unit=str(prediction.window.unit),
            position=prediction.window.position,
            ends_at=prediction.window.ends_at,
            target=prediction.target,
            predicted=prediction.predicted,
        )

    def to_prediction(self) -> WindowPrediction:
        return self.prediction_of(
            self.unit, self.position, self.ends_at, self.target, self.predicted
        )

    @staticmethod
    def prediction_of(
        unit: str, position: int, ends_at: float, target: float, predicted: float
    ) -> WindowPrediction:
        """The answer a row's columns hold, for a row read without the mapped instance."""
        return WindowPrediction(
            window=TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at),
            target=target,
            predicted=predicted,
        )
