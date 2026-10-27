from enum import StrEnum


class DetectorFeatures(StrEnum):
    """What a detector reads a window as, before it measures how unlike the normal windows it is.

    Attributes:
        ENCODER_STATES: The states of a frozen encoder pooled over the window, pretrained or
            drawn at its initialisation, read at the layer and under the pooling a plan names.
        SUMMARIES: The statistics of every channel of the window, the table the classical
            methods read; the bar a representation is held to.
    """

    ENCODER_STATES = "encoder_states"
    SUMMARIES = "summaries"

    @property
    def reads_an_encoder(self) -> bool:
        return self is DetectorFeatures.ENCODER_STATES
