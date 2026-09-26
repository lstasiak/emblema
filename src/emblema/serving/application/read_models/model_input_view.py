"""What a served model takes, as a client preparing a request reads it."""

from dataclasses import dataclass
from typing import Self

from emblema.serving.domain.model_input import ModelInput


@dataclass(frozen=True, kw_only=True)
class InputChannelView:
    """One channel of the model's vocabulary, and whether a reading on it will be taken.

    Attributes:
        name: Name of the channel, as a request spells it.
        timeless: Whether it is a static feature of the window rather than a measurement.
        unit: Physical unit of its values, where the corpus documents one.
        known: Whether the model can take a reading on it; a channel the training data never
            observed is listed and refused, so a client knows why its reading was ignored.
    """

    name: str
    timeless: bool
    unit: str | None
    known: bool


@dataclass(frozen=True, kw_only=True)
class ModelInputView:
    """The corpus, window and channels a request is written against.

    Attributes:
        corpus: Name of the corpus the candidate was fitted to.
        window_length: Extent of the windows it was fitted on, in the corpus's unit of time.
        channels: The vocabulary, in identifier order.
    """

    corpus: str
    window_length: float
    channels: tuple[InputChannelView, ...]

    @classmethod
    def of(cls, model_input: ModelInput) -> Self:
        return cls(
            corpus=model_input.corpus,
            window_length=model_input.window_length,
            channels=tuple(
                InputChannelView(
                    name=channel.channel,
                    timeless=channel.timeless,
                    unit=channel.unit,
                    known=model_input.knows(channel.channel),
                )
                for channel in model_input.channels
            ),
        )
