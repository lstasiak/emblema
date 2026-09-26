"""What a served model takes, as a client preparing a request reads it: input and channels."""

from typing import Self

from pydantic import BaseModel, Field

from emblema.serving.application.read_models.model_input_view import ModelInputView


class InputChannelResource(BaseModel):
    """One channel of the model's vocabulary, and whether a reading on it will be taken."""

    name: str
    timeless: bool
    unit: str | None
    known: bool = Field(
        description="Whether the model takes readings on it; a channel never observed in "
        "training is listed and refused."
    )


class ModelInputResource(BaseModel):
    """The corpus, window and channels a request to the model is written against."""

    corpus: str
    window_length: float
    channels: list[InputChannelResource]

    @classmethod
    def of(cls, view: ModelInputView) -> Self:
        return cls(
            corpus=view.corpus,
            window_length=view.window_length,
            channels=[
                InputChannelResource(
                    name=channel.name,
                    timeless=channel.timeless,
                    unit=channel.unit,
                    known=channel.known,
                )
                for channel in view.channels
            ],
        )
