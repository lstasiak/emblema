from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError


@dataclass(frozen=True, kw_only=True)
class DetectorRecipe:
    """How a detector is made: what it reads a window as, and how it scores one from the rest.

    A detector is fitted on the windows of a task's fitting side without a label. One that reads
    features scores a window by a distance from the fitted windows, every feature standardised
    over the fitting side: the mean distance to the nearest of them, or the whitened distance
    from their mean; one that asks the pretext scores it by the error of what is filled in where
    a backbone's masks hide its cells. Its alarm is set where a stated share
    of the fitting windows, each scored as the rest are, would be flagged, so the score a run
    answers is the score above that alarm: positive is an alarm, and the alarm is set without a
    label. The score may first be read against the window's own unit, the alarm then set over
    the fitting windows read so.

    Invariants: an encoder is read exactly when the features are an encoder's states; a
    backbone's masks are named exactly when the pretext is asked; neighbours are named, at least
    one, exactly when the distance to the nearest fitted windows is the score; a detector scored
    by an error has no distance to choose; the contamination lies strictly between zero and
    one.

    Attributes:
        features: What a window is read as.
        reading: The encoder the states come from and how they are pooled, for features that
            are an encoder's; ``None`` for the summaries.
        neighbours: How many nearest fitted windows a score is the mean distance to; ``None``
            where the score is an error or the whitened distance.
        contamination: The share of fitting windows the alarm is set to flag.
        seed: Seed of the fit, which decides what an encoder at its initialisation draws.
        distance: Which distance from the fitted windows a detector over features scores by.
        reference: What a window's score is read against before the alarm.
        pretext: Whose masks hide the cells, and how often, for a detector that asks the
            pretext; ``None`` otherwise.
    """

    features: DetectorFeatures
    reading: EncoderReading | None
    neighbours: int | None
    contamination: float
    seed: int
    distance: ScoreDistance = ScoreDistance.NEAREST_NEIGHBOURS
    reference: ScoreReference = ScoreReference.FITTING_SIDE
    pretext: PretextAsking | None = None

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
        if (self.pretext is None) == self.features.asks_the_pretext:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} "
                + ("names no masks" if self.pretext is None else "names masks it does not ask")
            )
        if self.features.asks_the_pretext and self.distance is not ScoreDistance.NEAREST_NEIGHBOURS:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} scores by an error; it has no distance to choose"
            )
        if (self.neighbours is None) != (not self.scores_by_nearest_neighbours):
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} by the {self.distance} distance "
                + ("names no neighbours" if self.neighbours is None else "reads no neighbours")
            )
        if self.neighbours is not None and self.neighbours < 1:
            raise InvalidDetectorRecipeError(
                f"a score is the distance to at least one neighbour, got {self.neighbours}"
            )
        if not (isfinite(self.contamination) and 0.0 < self.contamination < 1.0):
            raise InvalidDetectorRecipeError(
                f"the contamination lies strictly between zero and one, got {self.contamination}"
            )

    @property
    def scores_by_nearest_neighbours(self) -> bool:
        """Whether the score is the mean distance to the nearest fitted windows."""
        return (
            not self.features.asks_the_pretext and self.distance is ScoreDistance.NEAREST_NEIGHBOURS
        )

    def nearest(self) -> int:
        """How many nearest fitted windows a distance is the mean to.

        Raises:
            InvalidDetectorRecipeError: If the detector scores by an error or by the whitened
                distance, which reads no neighbours.
        """
        if self.neighbours is None:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} scores by an error, not a distance"
                if self.features.asks_the_pretext
                else f"a detector by the {self.distance} distance reads no neighbours"
            )
        return self.neighbours

    def asked(self) -> PretextAsking:
        """Whose masks hide the cells, and how often.

        Raises:
            InvalidDetectorRecipeError: If the detector reads features, not the pretext.
        """
        if self.pretext is None:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} reads features, not the pretext"
            )
        return self.pretext
