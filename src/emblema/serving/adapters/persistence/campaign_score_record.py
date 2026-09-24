from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKeyConstraint, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from emblema.serving.adapters.persistence.orm import Base
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.campaign_score import CampaignScore


class CampaignScoreRecord(Base):
    """Row of ``serving.campaign_score``: one figure a campaign reported for a promotable artifact.

    A long table rather than a document, so a metric the campaigns start reporting needs no
    migration. The key is the figure's position in the announcement, since the budget that
    means every window has no count to write and a nullable column cannot be part of a key.
    """

    __tablename__ = "campaign_score"
    __table_args__ = (
        ForeignKeyConstraint(
            ["campaign_ref", "candidate"],
            ["promotable_artifact.campaign_ref", "promotable_artifact.candidate"],
            name="fk_campaign_score_campaign_ref",
            ondelete="CASCADE",
        ),
        CheckConstraint("budget IS NULL OR budget >= 1", name="budget_counts_windows"),
        CheckConstraint("repeats >= 1", name="repeated"),
    )

    campaign_ref: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    candidate: Mapped[str] = mapped_column(Text, primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric: Mapped[str] = mapped_column(Text)
    budget: Mapped[int | None] = mapped_column(Integer)
    value: Mapped[float] = mapped_column(Float)
    repeats: Mapped[int] = mapped_column(Integer)

    @classmethod
    def of(cls, origin: ArtifactOrigin, position: int, score: CampaignScore) -> Self:
        return cls(
            campaign_ref=origin.campaign.value,
            candidate=str(origin.candidate),
            position=position,
            metric=score.metric,
            budget=score.budget,
            value=score.value,
            repeats=score.repeats,
        )

    def to_score(self) -> CampaignScore:
        return CampaignScore(
            metric=self.metric, budget=self.budget, value=self.value, repeats=self.repeats
        )
