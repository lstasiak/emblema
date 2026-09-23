from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.evaluation.adapters.persistence.campaign_unit_error_record import (
    CampaignUnitErrorRecord,
)
from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CampaignCellRecord(Base):
    """Row of ``evaluation.campaign_cell``: one point of a grid that has run, with its errors.

    The cell's own coordinates are its key, so a cell recorded twice is refused by the database
    as well as by the aggregate, and what a campaign has left to run is what has no row here.
    The budget is text because the largest one has no count to write: a coordinate needs one
    spelling, and a nullable column cannot be part of a key.
    """

    __tablename__ = "campaign_cell"
    __table_args__ = (
        CheckConstraint("seconds >= 0", name="seconds_not_negative"),
        CheckConstraint(
            "num_nonnulls(artifact_key, artifact_algorithm, artifact_digest) IN (0, 3)",
            name="artifact_is_whole",
        ),
    )

    campaign_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("evaluation_campaign.id", ondelete="CASCADE"), primary_key=True
    )
    candidate: Mapped[str] = mapped_column(Text, primary_key=True)
    budget: Mapped[str] = mapped_column(Text, primary_key=True)
    seed: Mapped[int] = mapped_column(Integer, primary_key=True)
    seconds: Mapped[float] = mapped_column(Float)
    artifact_key: Mapped[str | None] = mapped_column(Text)
    artifact_algorithm: Mapped[str | None] = mapped_column(Text)
    artifact_digest: Mapped[str | None] = mapped_column(Text)
    errors: Mapped[list[CampaignUnitErrorRecord]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=CampaignUnitErrorRecord.unit
    )

    @classmethod
    def from_result(cls, campaign_id: CampaignId, result: CellResult) -> Self:
        artifact = result.artifact
        return cls(
            campaign_id=campaign_id.value,
            candidate=str(result.cell.candidate),
            budget=result.cell.budget.text(),
            seed=result.cell.seed,
            seconds=result.seconds,
            artifact_key=None if artifact is None else artifact.key,
            artifact_algorithm=None if artifact is None else str(artifact.checksum.algorithm),
            artifact_digest=None if artifact is None else artifact.checksum.digest,
            errors=[
                CampaignUnitErrorRecord.of(campaign_id, result.cell, error)
                for error in result.errors
            ],
        )

    def to_result(self) -> CellResult:
        return CellResult(
            cell=CampaignCell(
                candidate=CandidateRef(self.candidate),
                budget=LabelBudget.parse(self.budget),
                seed=self.seed,
            ),
            errors=tuple(record.to_error() for record in self.errors),
            seconds=self.seconds,
            artifact=self._artifact(),
        )

    def _artifact(self) -> ArtifactRef | None:
        if (
            self.artifact_key is None
            or self.artifact_algorithm is None
            or self.artifact_digest is None
        ):
            return None
        return ArtifactRef(
            self.artifact_key,
            Checksum(HashAlgorithm(self.artifact_algorithm), self.artifact_digest),
        )
