from dataclasses import dataclass

from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.application.read_models.predicted_window import PredictedWindow
from emblema.serving.application.read_models.predictions import Predictions
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.serving.ports.served_model_repository import ServedModelRepository


@dataclass(frozen=True, kw_only=True)
class PredictWindowsQuery:
    """Request for a served model's answer over windows of raw readings.

    Attributes:
        served_model: Which model answers.
        windows: What to answer, each on the caller's own time axis.
    """

    served_model: ServedModelId
    windows: tuple[ObservedWindow, ...]


class PredictWindows:
    """Answers windows with a served model, over the channels it knows, and says what was dropped.

    The model is looked up and held to being in service; the windows are held to what it takes
    and the service allows, tokenised as its corpus was, and answered in one call to the
    runtime. What a client gets back is one answer per window in the order asked, each beside
    the channels it stands on and the ones it does not.
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

    def __call__(self, query: PredictWindowsQuery) -> Predictions:
        """The answers, in the order the windows were given.

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
            NonFiniteAnswerError: If the model answers a window with a non-finite value.
        """
        model = self._served.get(query.served_model)
        model.must_be_serving()
        prepared = self._admission.prepare(model, query.windows)
        answers = self._runtime.predict(model.artifact, [window.tokens for window in prepared])
        return Predictions(
            served_model=model.served_model_id,
            windows=tuple(
                PredictedWindow(
                    prediction=answer,
                    channels_used=window.admitted.used,
                    channels_ignored=window.admitted.ignored,
                    warnings=window.warnings,
                )
                for window, answer in zip(prepared, answers, strict=True)
            ),
        )
