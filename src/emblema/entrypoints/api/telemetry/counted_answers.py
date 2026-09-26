from collections.abc import Callable

from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.serving.application.read_models.embeddings import Embeddings
from emblema.serving.application.read_models.predictions import Predictions


class CountedAnswers[Q, R: (Predictions, Embeddings)]:
    """A use case that answers windows, with what it answered counted.

    Wraps the use case as the driving port it is — a callable from its query to its result —
    so the routes call it unchanged and the use case knows nothing of the count.
    """

    def __init__(self, answer: Callable[[Q], R], telemetry: Telemetry, *, kind: str) -> None:
        self._answer: Callable[[Q], R] = answer
        self._telemetry = telemetry
        self._kind = kind

    def __call__(self, query: Q) -> R:
        result = self._answer(query)
        attributes = {"served_model_id": str(result.served_model), "answer": self._kind}
        self._telemetry.windows_answered.add(len(result.windows), attributes)
        ignored = sum(len(window.channels_ignored) for window in result.windows)
        if ignored:
            self._telemetry.channels_ignored.add(ignored, attributes)
        return result
