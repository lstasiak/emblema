from dataclasses import dataclass

from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.application.read_models.embedded_window import EmbeddedWindow
from emblema.serving.application.read_models.embeddings import Embeddings
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.serving.ports.served_model_repository import ServedModelRepository


@dataclass(frozen=True, kw_only=True)
class EmbedWindowsQuery:
    """Request for a served model's representation of windows of raw readings.

    Attributes:
        served_model: Which model represents them.
        windows: What to represent, each on the caller's own time axis.
    """

    served_model: ServedModelId
    windows: tuple[ObservedWindow, ...]


class EmbedWindows:
    """Represents windows with a served model over the channels it knows, naming what was dropped.

    The same path as a prediction up to the runtime, so a representation and an answer of one
    window stand on the same tokens; what differs is what the runtime is asked for, and that a
    candidate with no representation to hand out refuses here by name.
    """

    def __init__(
        self,
        served: ServedModelRepository,
        runtime: InferenceRuntime,
        admission: WindowAdmission,
    ) -> None:
        self._served = served
        self._runtime = runtime
        self._admission = admission

    def __call__(self, query: EmbedWindowsQuery) -> Embeddings:
        """The representations, in the order the windows were given.

        Raises:
            ServedModelNotFoundError: If no model is stored under that identity.
            ServedModelNotServingError: If the model has been withdrawn.
            EmptyRequestError: If there is no window.
            TooManyWindowsError: If there are more than the service admits at once.
            UnobservedWindowError: If a window holds no reading the model takes.
            UntokenisableRequestError: If a reading is of the wrong kind for its channel.
            WindowTooLongError: If a window tokenises to more tokens than the service admits.
            ArtifactUnavailableError: If the model's artifact is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate the runtime reads.
            UnservableArtifactError: If the candidate is kept in no form the runtime runs.
            EmbeddingUnavailableError: If the candidate has no representation to hand out.
            NonFiniteAnswerError: If the model represents a window with a non-finite value.
        """
        model = self._served.get(query.served_model)
        model.must_be_serving()
        prepared = self._admission.prepare(model, query.windows)
        embeddings = self._runtime.embed(model.artifact, [window.tokens for window in prepared])
        return Embeddings(
            served_model=model.served_model_id,
            windows=tuple(
                EmbeddedWindow(
                    embedding=embedding,
                    channels_used=window.admitted.used,
                    channels_ignored=window.admitted.ignored,
                    warnings=window.warnings,
                )
                for window, embedding in zip(prepared, embeddings, strict=True)
            ),
        )
