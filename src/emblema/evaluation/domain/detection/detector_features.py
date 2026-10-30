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
        PRETEXT_ERROR: The error of the values a pretrained backbone's pretext fills in where
            its own masks hide the window's cells.
        LINEAR_ANSWER_ERROR: The error of the values a ridge per channel fills in from the same
            visible cells under the same masks, fitted on the fitting side: the answer that
            learnt nothing from the backbone, held to what the backbone sees.
        INTERPOLATION_ERROR: The error of the values each channel's visible bins interpolate
            under the same masks, beside the ridge.
    """

    ENCODER_STATES = "encoder_states"
    SUMMARIES = "summaries"
    READING_COUNTS = "reading_counts"
    PRETEXT_ERROR = "pretext_error"
    LINEAR_ANSWER_ERROR = "linear_answer_error"
    INTERPOLATION_ERROR = "interpolation_error"

    @property
    def reads_an_encoder(self) -> bool:
        """Whether a window is read as the states of an encoder."""
        return self is DetectorFeatures.ENCODER_STATES

    @property
    def asks_the_pretext(self) -> bool:
        """Whether a window is scored by the error of what is filled in under a backbone's masks."""
        return self in (
            DetectorFeatures.PRETEXT_ERROR,
            DetectorFeatures.LINEAR_ANSWER_ERROR,
            DetectorFeatures.INTERPOLATION_ERROR,
        )
