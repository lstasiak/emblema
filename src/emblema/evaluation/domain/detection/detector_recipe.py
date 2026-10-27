from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError


@dataclass(frozen=True, kw_only=True)
class DetectorRecipe:
    """How a detector is made: what it reads a window as, and how it scores one from the rest.

    A detector is fitted on the windows of a task's fitting side without a label, and scores a
    window by its mean distance to the nearest fitted windows, every feature standardised over
    the fitting side. Its alarm is set where a stated share of the fitting windows, each scored
    against the others, would be flagged, so the score a run answers is the distance above that
    alarm: positive is an alarm, and the alarm is set without a label.

    Invariants: an encoder is read exactly when the features are an encoder's states; the
    neighbours are at least one; the contamination lies strictly between zero and one.

    Attributes:
        features: What a window is read as.
        reading: The encoder the states come from and how they are pooled, for features that
            are an encoder's; ``None`` for the summaries.
        neighbours: How many nearest fitted windows a score is the mean distance to.
        contamination: The share of fitting windows the alarm is set to flag.
        seed: Seed of the fit, which decides what an encoder at its initialisation draws.
    """

    features: DetectorFeatures
    reading: EncoderReading | None
    neighbours: int
    contamination: float
    seed: int

    def __post_init__(self) -> None:
        if (self.reading is None) == self.features.reads_an_encoder:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} "
                + (
                    "names no encoder"
                    if self.reading is None
                    else "names an encoder it does not read"
                )
            )
        if self.neighbours < 1:
            raise InvalidDetectorRecipeError(
                f"a score is the distance to at least one neighbour, got {self.neighbours}"
            )
        if not (isfinite(self.contamination) and 0.0 < self.contamination < 1.0):
            raise InvalidDetectorRecipeError(
                f"the contamination lies strictly between zero and one, got {self.contamination}"
            )
