from dataclasses import dataclass

from emblema.serving.application.read_models.model_input_view import ModelInputView
from emblema.serving.application.read_models.served_model_summary import ServedModelSummary


@dataclass(frozen=True, kw_only=True)
class ServedModelView:
    """One served model in full: what a list shows of it, and what it takes as input.

    The input is read off the artifact and costs a read of the store, which is why a list of
    models leaves it out and one model carries it.

    Attributes:
        summary: Identity, state, provenance and the artifact served.
        input: The corpus, window and channels a request is written against.
    """

    summary: ServedModelSummary
    input: ModelInputView
