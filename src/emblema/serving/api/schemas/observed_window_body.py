"""A window of raw readings as a request carries it: the window and the two kinds of reading."""

from pydantic import BaseModel, Field

from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.static_value import StaticValue


class ObservedValueBody(BaseModel):
    """One reading of one channel at one instant."""

    channel: str = Field(description="Name of the channel, as the model's input lists it.")
    time: float = Field(description="When it was observed, on the caller's own time axis.")
    value: float = Field(description="What was observed, in the channel's own unit.")


class StaticValueBody(BaseModel):
    """One value that holds for the whole window."""

    channel: str = Field(description="Name of a timeless channel, as the model's input lists it.")
    value: float = Field(description="The value, in the channel's own unit.")


class ObservedWindowBody(BaseModel):
    """One window of readings: where it starts, how long it is, and what was read inside it."""

    start: float = Field(description="First instant of the window, on the caller's time axis.")
    length: float = Field(description="Extent of the window, in the corpus's unit of time.")
    observations: list[ObservedValueBody] = Field(
        min_length=1, description="Timed readings inside the window, in any order."
    )
    static_features: list[StaticValueBody] = Field(
        default_factory=list, description="Values that hold for the whole window."
    )

    def to_message(self) -> ObservedWindow:
        """The window as the Catalog's service takes it.

        Raises:
            InvalidObservedWindowError: If a reading falls outside the window, or the window
                breaks another of its rules.
        """
        return ObservedWindow(
            start=self.start,
            length=self.length,
            observations=tuple(
                ObservedValue(channel=r.channel, time=r.time, value=r.value)
                for r in self.observations
            ),
            static_features=tuple(
                StaticValue(channel=f.channel, value=f.value) for f in self.static_features
            ),
        )
