from collections.abc import Iterable
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.evaluation.adapters.persistence.campaign_unit_error_record import (
    CampaignUnitErrorRecord,
)
from emblema.evaluation.adapters.persistence.campaign_window_prediction_record import (
    CampaignWindowPredictionRecord,
)
from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.cell_result import CellResult
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class CampaignCellRecord(Base):
    """Row of ``evaluation.campaign_cell``: one point of a grid that has run, with its results.

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
    predictions: Mapped[list[CampaignWindowPredictionRecord]] = relationship(
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=(CampaignWindowPredictionRecord.unit, CampaignWindowPredictionRecord.position),
    )

    @staticmethod
    def key_of(cell: CampaignCell) -> tuple[str, str, int]:
        """The cell's coordinates as the row spells them, without its campaign."""
        return str(cell.candidate), cell.budget.text(), cell.seed

    @staticmethod
    def cell_of(candidate: str, budget: str, seed: int) -> CampaignCell:
        """The cell a row's coordinates name."""
        return CampaignCell(
            candidate=CandidateRef(candidate), budget=LabelBudget.parse(budget), seed=seed
        )

    @classmethod
    def from_result(cls, campaign_id: CampaignId, result: CellResult) -> Self:
        artifact = result.artifact
        candidate, budget, seed = cls.key_of(result.cell)
        return cls(
            campaign_id=campaign_id.value,
            candidate=candidate,
            budget=budget,
            seed=seed,
            seconds=result.seconds,
            artifact_key=None if artifact is None else artifact.key,
            artifact_algorithm=None if artifact is None else str(artifact.checksum.algorithm),
            artifact_digest=None if artifact is None else artifact.checksum.digest,
            errors=[
                CampaignUnitErrorRecord.of(campaign_id, result.cell, error)
                for error in result.errors
            ],
            predictions=[
                CampaignWindowPredictionRecord.of(campaign_id, result.cell, prediction)
                for prediction in result.predictions
            ],
        )

    def to_result(self) -> CellResult:
        return self.result_of(
            self.candidate,
            self.budget,
            self.seed,
            self.seconds,
            self.artifact_key,
            self.artifact_algorithm,
            self.artifact_digest,
            [record.to_error() for record in self.errors],
            [record.to_prediction() for record in self.predictions],
        )

    @classmethod
    def result_of(
        cls,
        candidate: str,
        budget: str,
        seed: int,
        seconds: float,
        artifact_key: str | None,
        artifact_algorithm: str | None,
        artifact_digest: str | None,
        errors: Iterable[UnitError],
        predictions: Iterable[WindowPrediction],
    ) -> CellResult:
        """The result a cell's row holds with its errors and answers, read without the instances."""
        artifact = (
            None
            if artifact_key is None or artifact_algorithm is None or artifact_digest is None
            else ArtifactRef(
                artifact_key, Checksum(HashAlgorithm(artifact_algorithm), artifact_digest)
            )
        )
        return CellResult(
            cell=cls.cell_of(candidate, budget, seed),
            errors=tuple(errors),
            seconds=seconds,
            artifact=artifact,
            predictions=tuple(predictions),
        )
