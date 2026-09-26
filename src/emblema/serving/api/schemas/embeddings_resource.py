"""What a served model represented windows as: every window's representation, and all of them."""

from typing import Self

from pydantic import BaseModel

from emblema.serving.application.read_models.embeddings import Embeddings


class EmbeddedWindowResource(BaseModel):
    """The representation of one window, beside what it stands on and what was left out."""

    embedding: list[float]
    channels_used: list[str]
    channels_ignored: list[str]
    warnings: list[str]


class EmbeddingsResource(BaseModel):
    """What a served model represented the windows as, in the order asked."""

    served_model_id: str
    windows: list[EmbeddedWindowResource]

    @classmethod
    def of(cls, embeddings: Embeddings) -> Self:
        return cls(
            served_model_id=str(embeddings.served_model),
            windows=[
                EmbeddedWindowResource(
                    embedding=list(window.embedding),
                    channels_used=list(window.channels_used),
                    channels_ignored=list(window.channels_ignored),
                    warnings=list(window.warnings),
                )
                for window in embeddings.windows
            ],
        )
