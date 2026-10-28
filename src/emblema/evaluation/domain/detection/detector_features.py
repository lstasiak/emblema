from enum import StrEnum


class DetectorFeatures(StrEnum):
    """What a detector reads a window as, before it measures how unlike the normal windows it is.

    Attributes:
        ENCODER_STATES: The states of a frozen encoder pooled over the window, pretrained or
            drawn at its initialisation, read at the layer and under the pooling a plan names.
        SUMMARIES: The statistics of every channel of the window, the table the classical
            methods read; the bar a representation is held to.
        READING_COUNTS: How many readings each channel of the window holds, nothing of their
            values: what the sampling alone tells, the bar a detector has to pass to show it
            reads more than when the telemetry was sent.
    """

    ENCODER_STATES = "encoder_states"
    SUMMARIES = "summaries"
    READING_COUNTS = "reading_counts"

    @property
    def reads_an_encoder(self) -> bool:
        return self is DetectorFeatures.ENCODER_STATES
