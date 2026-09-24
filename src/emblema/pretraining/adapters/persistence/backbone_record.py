from datetime import datetime
from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.pretraining.adapters.documents.experiment_configuration_document import (
    ExperimentConfigurationDocument,
)
from emblema.pretraining.adapters.persistence.orm import Base
from emblema.pretraining.adapters.persistence.pretraining_input_record import (
    PretrainingInputRecord,
)
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

ORDERED, READY = "ordered", "ready"
CONFIGURATIONS = ExperimentConfigurationDocument()


class BackboneRecord(Base):
    """Row of ``pretraining.backbone`` with its inputs: the persistence model of the aggregate.

    The configuration is JSONB, a document whose shape changes more often than it is queried;
    the seed, the parameter count and the commit are columns beside it because they are asked
    about and a query should not have to open the document. The status is a column of its own,
    so that what is ready can be selected without knowing which columns a delivery fills, and
    check constraints keep it true to them: a ready row names its weights and the result they
    came with, and is dated.
    """

    __tablename__ = "backbone"
    __table_args__ = (
        CheckConstraint(f"status IN ('{ORDERED}', '{READY}')", name="status_known"),
        CheckConstraint(
            f"(status = '{READY}') = (artifact_digest IS NOT NULL)", name="ready_has_artifact"
        ),
        CheckConstraint(
            "(artifact_digest IS NULL) = (delivered_at IS NULL)", name="delivery_dated"
        ),
        CheckConstraint(
            "(artifact_digest IS NULL) = (result_digest IS NULL)", name="delivery_has_result"
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    experiment: Mapped[str] = mapped_column(Text)
    run: Mapped[str] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(Text)
    configuration: Mapped[dict[str, object]] = mapped_column(JSONB)
    parameter_count: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    git_commit: Mapped[str] = mapped_column(Text)
    signature: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    result_key: Mapped[str | None] = mapped_column(Text)
    result_algorithm: Mapped[str | None] = mapped_column(Text)
    result_digest: Mapped[str | None] = mapped_column(Text)
    artifact_key: Mapped[str | None] = mapped_column(Text)
    artifact_algorithm: Mapped[str | None] = mapped_column(Text)
    artifact_digest: Mapped[str | None] = mapped_column(Text)
    ordered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    inputs: Mapped[list[PretrainingInputRecord]] = relationship(
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=PretrainingInputRecord.position,
    )

    @classmethod
    def from_backbone(cls, backbone: Backbone) -> Self:
        result, artifact = backbone.result, backbone.artifact
        return cls(
            id=backbone.id.value,
            experiment=backbone.configuration.name,
            run=backbone.run,
            tier=str(backbone.configuration.tier),
            configuration=CONFIGURATIONS.encode(backbone.configuration),
            parameter_count=backbone.parameter_count,
            seed=backbone.seed,
            git_commit=backbone.git_commit,
            signature=backbone.signature.digest,
            status=READY if backbone.is_ready else ORDERED,
            result_key=None if result is None else result.key,
            result_algorithm=None if result is None else str(result.checksum.algorithm),
            result_digest=None if result is None else result.checksum.digest,
            artifact_key=None if artifact is None else artifact.key,
            artifact_algorithm=None if artifact is None else str(artifact.checksum.algorithm),
            artifact_digest=None if artifact is None else artifact.checksum.digest,
            ordered_at=backbone.ordered_at.value,
            delivered_at=None if backbone.delivered_at is None else backbone.delivered_at.value,
            inputs=[
                PretrainingInputRecord.from_input(backbone.id, position, read)
                for position, read in enumerate(backbone.inputs)
            ],
        )

    def to_backbone(self) -> Backbone:
        return Backbone(
            id=BackboneId(self.id),
            configuration=CONFIGURATIONS.decode(self.configuration),
            inputs=tuple(read.to_input() for read in self.inputs),
            run=self.run,
            git_commit=self.git_commit,
            signature=RunSignature(self.signature),
            ordered_at=as_utc(self.ordered_at),
            result=self._ref(self.result_key, self.result_algorithm, self.result_digest),
            artifact=self._ref(self.artifact_key, self.artifact_algorithm, self.artifact_digest),
            delivered_at=None if self.delivered_at is None else as_utc(self.delivered_at),
        )

    @staticmethod
    def _ref(key: str | None, algorithm: str | None, digest: str | None) -> ArtifactRef | None:
        if key is None or algorithm is None or digest is None:
            return None
        return ArtifactRef(key, Checksum(HashAlgorithm(algorithm), digest))
