from dataclasses import dataclass

from emblema.serving.application.read_models.predicted_window import PredictedWindow
from emblema.serving.domain.identifiers import ServedModelId


@dataclass(frozen=True, kw_only=True)
class Predictions:
    """What a served model answered for a request, one answer per window, in the order asked.

    Attributes:
        served_model: The model that answered.
        windows: The answers, in the order the request stated its windows.
    """

    served_model: ServedModelId
    windows: tuple[PredictedWindow, ...]
