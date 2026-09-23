from pydantic import BaseModel, Field


class ScheduleSettings(BaseModel):
    """How long every cell of this worker's campaigns learns, and in how large a step.

    Nothing has a default. How much a run spends and how fast it learns are what the comparison
    is about, so a worker that was told neither must refuse to start rather than measure a
    schedule nobody declared. What each field means is ``AdaptationSchedule``'s to say.
    """

    epochs: int
    min_steps: int = Field(description="Steps a run takes at least, whatever its sample holds.")
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_fraction: float = Field(description="Share of the run's steps the rate climbs over.")
    final_lr_fraction: float = Field(description="Fraction of the peak the rate decays to.")
