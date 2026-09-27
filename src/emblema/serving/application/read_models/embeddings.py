from dataclasses import dataclass

from emblema.serving.application.read_models.embedded_window import EmbeddedWindow
from emblema.serving.domain.identifiers import ServedModelId


@dataclass(frozen=True, kw_only=True)
class Embeddings:
    """What a served model represented a request's windows as, in the order asked.

    Attributes:
        served_model: The model that answered.
        windows: The representations, in the order the request stated its windows.
    """

    served_model: ServedModelId
    windows: tuple[EmbeddedWindow, ...]
