from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Index, Text, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef, TaskId
from emblema.serving.adapters.persistence.orm import Base
from emblema.serving.domain.artifact_origin import ArtifactOrigin
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel
from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

SERVING_ARTIFACT = "uq_served_model_serving_artifact"


class ServedModelRecord(Base):
    """Row of ``serving.served_model``: one period of service of one artifact.

    The origin is copied rather than joined to the projection it was promoted out of: a model
    and the projection are separate aggregates, and a model has to go on saying where its
    artifact came from whatever later becomes of the projection. That one artifact is served by
    at most one model at a time is a partial unique index over the models not withdrawn, so two
    processes promoting it at once cannot both succeed.
    """

    __tablename__ = "served_model"
    __table_args__ = (
        CheckConstraint(
            f"kind IN ('{CandidateKind.NEURAL}', '{CandidateKind.CLASSICAL}')", name="kind_known"
        ),
        CheckConstraint(
            "withdrawn_at IS NULL OR withdrawn_at >= promoted_at", name="withdrawn_after_promoted"
        ),
        Index(
            SERVING_ARTIFACT,
            "artifact_algorithm",
            "artifact_digest",
            unique=True,
            postgresql_where=text("withdrawn_at IS NULL"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    campaign_ref: Mapped[UUID] = mapped_column(Uuid)
    task_ref: Mapped[UUID] = mapped_column(Uuid)
    candidate: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    artifact_key: Mapped[str] = mapped_column(Text)
    artifact_algorithm: Mapped[str] = mapped_column(Text)
    artifact_digest: Mapped[str] = mapped_column(Text)
    promoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @classmethod
    def from_model(cls, model: ServedModel) -> Self:
        return cls(
            id=model.served_model_id.value,
            campaign_ref=model.origin.campaign.value,
            task_ref=model.origin.task.value,
            candidate=str(model.origin.candidate),
            kind=str(model.kind),
            artifact_key=model.artifact.key,
            artifact_algorithm=str(model.artifact.checksum.algorithm),
            artifact_digest=model.artifact.checksum.digest,
            promoted_at=model.promoted_at.value,
            withdrawn_at=None if model.withdrawn_at is None else model.withdrawn_at.value,
        )

    def to_model(self) -> ServedModel:
        return ServedModel(
            served_model_id=ServedModelId(self.id),
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
            promoted_at=as_utc(self.promoted_at),
            withdrawn_at=None if self.withdrawn_at is None else as_utc(self.withdrawn_at),
        )
