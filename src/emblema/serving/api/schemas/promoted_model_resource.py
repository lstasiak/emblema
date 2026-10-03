from typing import Self

from pydantic import BaseModel, Field

from emblema.serving.domain.identifiers import ServedModelId


class PromotedModelResource(BaseModel):
    """The model now serving a promoted artifact, by identity; its detail is a read away."""

    served_model_id: str = Field(description="The identity of the model now serving the artifact.")

    @classmethod
    def of(cls, served_model_id: ServedModelId) -> Self:
        return cls(served_model_id=str(served_model_id))
