from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from emblema.serving.adapters.persistence.served_model_record import (
    SERVING_ARTIFACT,
    ServedModelRecord,
)
from emblema.serving.domain.exceptions import ArtifactAlreadyServedError, ServedModelNotFoundError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel


class SqlAlchemyServedModelRepository:
    """Repository over the Serving schema of the metadata database.

    A model is one row, merged by identity. That an artifact is served once at a time is the
    partial unique index of the schema, surfaced as the domain's error; any other integrity
    failure is a bug and propagates.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get(self, served_model_id: ServedModelId) -> ServedModel:
        with Session(self._engine) as session:
            record = session.get(ServedModelRecord, served_model_id.value)
            if record is None:
                raise ServedModelNotFoundError(f"no served model {served_model_id}")
            return record.to_model()

    def save(self, model: ServedModel) -> None:
        try:
            with Session(self._engine) as session, session.begin():
                session.merge(ServedModelRecord.from_model(model))
        except IntegrityError as error:
            if self._serves_twice(error):
                raise ArtifactAlreadyServedError(
                    f"artifact {model.artifact.checksum} is already served"
                ) from error
            raise

    @staticmethod
    def _serves_twice(error: IntegrityError) -> bool:
        diagnostics = getattr(error.orig, "diag", None)
        return getattr(diagnostics, "constraint_name", None) == SERVING_ARTIFACT
