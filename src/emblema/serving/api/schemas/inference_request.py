from pydantic import BaseModel, Field

from emblema.serving.api.schemas.observed_window_body import ObservedWindowBody


class InferenceRequest(BaseModel):
    """Windows to answer, each on the caller's own time axis."""

    windows: list[ObservedWindowBody] = Field(min_length=1)
