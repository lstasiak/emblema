"""What a served model answered: every window's answer, and the answers together."""

from typing import Self

from pydantic import BaseModel, Field

from emblema.serving.application.read_models.predictions import Predictions


class PredictedWindowResource(BaseModel):
    """The answer for one window, beside what it stands on and what was left out."""

    prediction: float = Field(description="The candidate's answer, in the task's unit.")
    channels_used: list[str]
    channels_ignored: list[str] = Field(
        description="Channels the request read that the model does not know; ignored, not refused."
    )
    warnings: list[str]


class PredictionsResource(BaseModel):
    """What a served model answered, one answer per window, in the order asked."""

    served_model_id: str
    windows: list[PredictedWindowResource]

    @classmethod
    def of(cls, predictions: Predictions) -> Self:
        return cls(
            served_model_id=str(predictions.served_model),
            windows=[
                PredictedWindowResource(
                    prediction=window.prediction,
                    channels_used=list(window.channels_used),
                    channels_ignored=list(window.channels_ignored),
                    warnings=list(window.warnings),
                )
                for window in predictions.windows
            ],
        )
