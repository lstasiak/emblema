from dataclasses import dataclass

from emblema.serving.application.read_models.model_input_view import ModelInputView
from emblema.serving.application.read_models.served_model_summary import ServedModelSummary
from emblema.serving.application.read_models.served_model_view import ServedModelView
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.serving.ports.served_model_repository import ServedModelRepository


@dataclass(frozen=True, kw_only=True)
class ViewServedModelQuery:
    """Which served model to show.

    Attributes:
        served_model: Identity of the model.
    """

    served_model: ServedModelId


class ViewServedModel:
    """Shows one served model with what it takes, withdrawn or not.

    A withdrawn model is still shown: what served, and from when to when, is the provenance of
    every answer it gave, and a client holding such an answer may ask what gave it.
    """

    def __init__(self, served: ServedModelRepository, runtime: InferenceRuntime) -> None:
        self._served = served
        self._runtime = runtime

    def __call__(self, query: ViewServedModelQuery) -> ServedModelView:
        """The model and its input.

        Raises:
            ServedModelNotFoundError: If no model is stored under that identity.
            ArtifactUnavailableError: If the model's artifact is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate the runtime reads.
        """
        model = self._served.get(query.served_model)
        return ServedModelView(
            summary=ServedModelSummary.of(model),
            input=ModelInputView.of(self._runtime.describe(model.artifact)),
        )
