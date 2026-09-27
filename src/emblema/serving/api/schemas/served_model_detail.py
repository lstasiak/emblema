from typing import Self

from pydantic import BaseModel

from emblema.serving.api.schemas.model_input_resource import ModelInputResource
from emblema.serving.api.schemas.served_model_resource import ServedModelResource
from emblema.serving.application.read_models.served_model_view import ServedModelView


class ServedModelDetail(BaseModel):
    """One served model in full: the model and what it takes as input."""

    model: ServedModelResource
    input: ModelInputResource

    @classmethod
    def of(cls, view: ServedModelView) -> Self:
        return cls(
            model=ServedModelResource.of(view.summary), input=ModelInputResource.of(view.input)
        )
