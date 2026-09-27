from collections.abc import Sequence
from math import isclose

from emblema.catalog.contracts.exceptions import UntokenisableWindowError
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.window_tokeniser import WindowTokeniser
from emblema.serving.application.admission.prepared_window import PreparedWindow
from emblema.serving.domain.admitted_window import AdmittedWindow
from emblema.serving.domain.exceptions import UntokenisableRequestError
from emblema.serving.domain.inference_limits import InferenceLimits
from emblema.serving.domain.model_input import ModelInput
from emblema.serving.domain.served_model import ServedModel
from emblema.serving.ports.inference_runtime import InferenceRuntime


class WindowAdmission:
    """Holds a request to what the model takes and the service allows, and tokenises it.

    The one path from raw readings to tokens for both things a model answers, so a prediction
    and a representation of the same window are computed over the same tokens. The policy for
    a reading the model cannot take is the model input's; the tokenisation is the Catalog's;
    what is refused outright — too many windows, a window too long, a reading of the wrong
    kind — is the service's. A window shorter or longer than the candidate was fitted on is
    answered and remarked on: the arithmetic runs, and the caller is told the ground moved.

    Every reading becomes one token, so a window too long is refused by counting its readings
    before any of them is tokenised: the tokeniser is the costly step, and a request that will
    be refused should not pay for it.
    """

    def __init__(
        self, runtime: InferenceRuntime, tokeniser: WindowTokeniser, limits: InferenceLimits
    ) -> None:
        self._runtime = runtime
        self._tokeniser = tokeniser
        self._limits = limits

    def prepare(
        self, model: ServedModel, windows: Sequence[ObservedWindow]
    ) -> tuple[PreparedWindow, ...]:
        """Every window of the request as the model is fed it, in the order given.

        Raises:
            EmptyRequestError: If there is no window.
            TooManyWindowsError: If there are more than the service admits at once.
            UnobservedWindowError: If a window holds no reading the model takes.
            UntokenisableRequestError: If a reading is of the wrong kind for its channel.
            WindowTooLongError: If a window holds more readings than the service tokenises.
            ArtifactUnavailableError: If the model's artifact is not in the store.
            UnreadableServedArtifactError: If it is not a kept candidate the runtime reads.
        """
        self._limits.admit_count(len(windows))
        model_input = self._runtime.describe(model.artifact)
        return tuple(self._prepared(model_input, window) for window in windows)

    def _prepared(self, model_input: ModelInput, window: ObservedWindow) -> PreparedWindow:
        admitted = model_input.admit(window)
        self._limits.admit_length(
            len(admitted.window.observations) + len(admitted.window.static_features)
        )
        try:
            tokens = self._tokeniser.tokenise(
                admitted.window, model_input.corpus, model_input.channels
            )
        except UntokenisableWindowError as error:
            raise UntokenisableRequestError(str(error)) from error
        return PreparedWindow(
            admitted=admitted, tokens=tokens, warnings=self._warnings(model_input, admitted)
        )

    @staticmethod
    def _warnings(model_input: ModelInput, admitted: AdmittedWindow) -> tuple[str, ...]:
        warnings = []
        if admitted.ignored:
            warnings.append(
                "readings on channels the model does not know were ignored: "
                + ", ".join(admitted.ignored)
            )
        length = admitted.window.length
        if not isclose(length, model_input.window_length):
            warnings.append(
                f"the window spans {length:g} and the candidate was fitted on windows of "
                f"{model_input.window_length:g}"
            )
        return tuple(warnings)
