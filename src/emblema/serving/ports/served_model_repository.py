from typing import Protocol

from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model import ServedModel


class ServedModelRepository(Protocol):
    """Keeps served models, withdrawn ones included, as the record of what answered and when.

    One artifact is served by at most one model at a time. That is a rule over every stored
    model rather than over one, so it is the repository that holds it, where two processes
    promoting the same artifact at once cannot both pass it.
    """

    def get(self, served_model_id: ServedModelId) -> ServedModel:
        """The model stored under that identity.

        Raises:
            ServedModelNotFoundError: If no model is stored under it.
        """
        ...

    def save(self, model: ServedModel) -> None:
        """Store the model, replacing any earlier state of it.

        Raises:
            ArtifactAlreadyServedError: If the model serves an artifact that another model,
                not withdrawn, already serves.
        """
        ...
