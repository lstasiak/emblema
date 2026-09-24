from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.candidate_standing import CandidateStanding
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.serving.adapters.persistence.campaign_score_record import CampaignScoreRecord
from emblema.serving.adapters.persistence.orm import Base
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class PromotableArtifactRecord(Base):
    """Row of ``serving.promotable_artifact`` with its scores: the projection as it is stored.

    The campaign and the task are referred to by identity alone, with no foreign key behind
    them: they live in another context's schema, and the promise that the campaign finished is
    kept by this projection having been built from its announcement, not by the database. The
    origin is the key, which is what makes a second delivery of one announcement replace the
    first rather than add to it. The checksum is indexed because it is what a promotion names.
    """

    __tablename__ = "promotable_artifact"
    __table_args__ = (
        CheckConstraint(
            f"kind IN ('{CandidateKind.NEURAL}', '{CandidateKind.CLASSICAL}')", name="kind_known"
        ),
        CheckConstraint(
            f"standing IN ('{CandidateStanding.CONTROL}', '{CandidateStanding.ESTABLISHED}', "
            f"'{CandidateStanding.NOT_ESTABLISHED}', '{CandidateStanding.WORSE}')",
            name="standing_known",
        ),
    )

    campaign_ref: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    candidate: Mapped[str] = mapped_column(Text, primary_key=True)
    task_ref: Mapped[UUID] = mapped_column(Uuid)
    kind: Mapped[str] = mapped_column(Text)
    standing: Mapped[str] = mapped_column(Text)
    artifact_key: Mapped[str] = mapped_column(Text)
    artifact_algorithm: Mapped[str] = mapped_column(Text)
    artifact_digest: Mapped[str] = mapped_column(Text, index=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    scores: Mapped[list[CampaignScoreRecord]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=CampaignScoreRecord.position
    )

    @classmethod
    def from_artifact(cls, artifact: PromotableArtifact) -> Self:
        origin = artifact.origin
        return cls(
            campaign_ref=origin.campaign.value,
            candidate=str(origin.candidate),
            task_ref=origin.task.value,
            kind=str(artifact.kind),
            standing=str(artifact.standing),
            artifact_key=artifact.artifact.key,
            artifact_algorithm=str(artifact.artifact.checksum.algorithm),
            artifact_digest=artifact.artifact.checksum.digest,
            completed_at=artifact.completed_at.value,
            scores=[
                CampaignScoreRecord.of(origin, position, score)
                for position, score in enumerate(artifact.scores)
            ],
        )

    def to_artifact(self) -> PromotableArtifact:
        return PromotableArtifact(
            origin=ArtifactOrigin(
                campaign=CampaignId(self.campaign_ref),
                task=TaskId(self.task_ref),
                candidate=CandidateRef(self.candidate),
            ),
            kind=CandidateKind(self.kind),
            artifact=ArtifactRef(
                self.artifact_key,
                Checksum(HashAlgorithm(self.artifact_algorithm), self.artifact_digest),
            ),
            standing=CandidateStanding(self.standing),
            scores=tuple(record.to_score() for record in self.scores),
            completed_at=as_utc(self.completed_at),
        )
