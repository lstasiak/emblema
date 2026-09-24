from emblema.serving.domain.exceptions import ArtifactAlreadyServedError, ServedModelNotFoundError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.domain.served_model_state import ServedModelState


class InMemoryServedModelRepository:
    """Keeps served models in a dictionary, checking on save what the database's index checks."""

    def __init__(self) -> None:
        self._models: dict[ServedModelId, ServedModel] = {}

    def get(self, served_model_id: ServedModelId) -> ServedModel:
        try:
            return self._models[served_model_id]
        except KeyError as error:
            raise ServedModelNotFoundError(f"no served model {served_model_id}") from error

    def save(self, model: ServedModel) -> None:
        if model.state is ServedModelState.SERVING and any(
            other.served_model_id != model.served_model_id
            and other.state is ServedModelState.SERVING
            and other.artifact.checksum == model.artifact.checksum
            for other in self._models.values()
        ):
            raise ArtifactAlreadyServedError(
                f"artifact {model.artifact.checksum} is already served"
            )
        self._models[model.served_model_id] = model
