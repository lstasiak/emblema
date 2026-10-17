from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from emblema.pretraining.adapters.persistence.backbone_record import BackboneRecord
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.exceptions import BackboneNotFoundError
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.shared.kernel.artifacts import ArtifactRef


class SqlAlchemyBackboneRepository:
    """Repository over the Pretraining schema of the metadata database.

    A backbone is read whole and written whole: ``save`` merges the record of the state the
    caller holds, so the row and its input are inserted or updated by identity in one
    transaction. Nothing here is unique but the identity, so no integrity failure has a domain
    meaning and any one propagates.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, backbone_id: BackboneId) -> Backbone:
        with Session(self._engine) as session:
            record = session.get(BackboneRecord, backbone_id.value)
            if record is None:
                raise BackboneNotFoundError(f"no backbone {backbone_id}")
            return record.to_backbone()

    def get_by_artifact(self, artifact: ArtifactRef) -> Backbone:
        with Session(self._engine) as session:
            record = session.scalars(
                select(BackboneRecord)
                .where(
                    BackboneRecord.artifact_key == artifact.key,
                    BackboneRecord.artifact_algorithm == str(artifact.checksum.algorithm),
                    BackboneRecord.artifact_digest == artifact.checksum.digest,
                )
                .order_by(BackboneRecord.ordered_at)
            ).first()
            if record is None:
                raise BackboneNotFoundError(f"no ready backbone holds {artifact.key}")
            return record.to_backbone()

    def save(self, backbone: Backbone) -> None:
        with Session(self._engine) as session, session.begin():
            session.merge(BackboneRecord.from_backbone(backbone))
