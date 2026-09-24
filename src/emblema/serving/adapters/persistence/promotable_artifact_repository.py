from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from emblema.serving.adapters.persistence.promotable_artifact_record import (
    PromotableArtifactRecord,
)
from emblema.serving.domain.promotable_artifact import PromotableArtifact
from emblema.shared.kernel.checksums import Checksum


class SqlAlchemyPromotableArtifactRepository:
    """Repository over the Serving schema of the metadata database.

    An artifact is read whole and written whole: ``save`` merges the record under its origin,
    so the row and its scores are inserted or replaced in one transaction. Nothing is unique
    but the origin, so no integrity failure has a domain meaning and any one propagates.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, artifact: PromotableArtifact) -> None:
        with Session(self._engine) as session, session.begin():
            session.merge(PromotableArtifactRecord.from_artifact(artifact))

    def find_by_checksum(self, checksum: Checksum) -> tuple[PromotableArtifact, ...]:
        with Session(self._engine) as session:
            records = session.scalars(
                select(PromotableArtifactRecord)
                .where(
                    PromotableArtifactRecord.artifact_algorithm == str(checksum.algorithm),
                    PromotableArtifactRecord.artifact_digest == checksum.digest,
                )
                .order_by(
                    PromotableArtifactRecord.completed_at,
                    PromotableArtifactRecord.campaign_ref,
                    # Code-point order, which is what the port promises whatever collation the
                    # database was created with.
                    PromotableArtifactRecord.candidate.collate("C"),
                )
            )
            return tuple(record.to_artifact() for record in records)
