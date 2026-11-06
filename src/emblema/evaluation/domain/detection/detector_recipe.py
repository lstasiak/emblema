from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError


@dataclass(frozen=True, kw_only=True)
class DetectorRecipe:
    """How a detector is made: what it reads a window as, and how it scores one from the rest.

    A detector is fitted on the windows of a task's fitting side, by default without a label. One
    that reads features scores a window by a distance from the fitted windows, every feature
    standardised over the fitting side: the mean distance to the nearest of them, or the whitened
    distance from their mean; one that asks the pretext scores it by the error of what is filled in
    where a backbone's masks hide its cells; one over its channels' history scores it by how far
    its channels lie from their own earlier windows. A detector over features may instead be taught
    a counted number of the fitting side's events, and then scores by a head the events set,
    replacing the distance. Its alarm is set where a stated share of the fitting windows, each
    scored as the rest are, would be flagged, so the score a run answers is the score above that
    alarm: positive is an alarm, and the alarm is set without a label. The score may first be
    read against the window's own unit, the alarm then set over the fitting windows read so.

    Invariants: an encoder is read exactly when the features are an encoder's states; a
    backbone's masks are named exactly when the pretext is asked; a history is named exactly
    when the channels' departures are read, and such a detector has no distance to choose and
    learns from no event; neighbours are named, at least
    one, exactly when the distance to the nearest fitted windows is the score; a detector scored
    by an error has no distance to choose, and only such a detector reads its errors per
    channel; a detector taught events reads features, names no distance but the default and no
    neighbours, and only such a detector chooses a head; the contamination lies strictly between
    zero and one.

    Attributes:
        features: What a window is read as.
        reading: The encoder the states come from and how they are pooled, for features that
            are an encoder's; ``None`` for the summaries.
        neighbours: How many nearest fitted windows a score is the mean distance to; ``None``
            where the score is an error or a distance other than the nearest neighbours'.
        contamination: The share of fitting windows the alarm is set to flag.
        seed: Seed of the fit, which decides what an encoder at its initialisation draws.
        distance: Which distance from the fitted windows a detector over features scores by.
        normalisation: How the cells' errors become the score of a detector scored by an error.
        reference: What a window's score is read against before the alarm.
        pretext: Whose masks hide the cells, and how often, for a detector that asks the
            pretext; ``None`` otherwise.
        events: How many of the fitting side's events the detector is taught; nought for one
            that learns from none.
        head: How a taught detector scores a window by its events.
        history: How far back each channel is read and how its departures make the score, for
            a detector over its channels' history; ``None`` otherwise.
    """

    features: DetectorFeatures
    reading: EncoderReading | None
    neighbours: int | None
    contamination: float
    seed: int
    distance: ScoreDistance = ScoreDistance.NEAREST_NEIGHBOURS
    normalisation: ErrorNormalisation = ErrorNormalisation.RAW
    reference: ScoreReference = ScoreReference.FITTING_SIDE
    pretext: PretextAsking | None = None
    events: int = 0
    head: TaughtHead = TaughtHead.DIRECTION
    history: ChannelHistory | None = None

    def __post_init__(self) -> None:
        if (self.history is None) == self.features.reads_channel_history:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} "
                + (
                    "names no history"
                    if self.history is None
                    else "names a history it does not read"
                )
            )
        if self.features.reads_channel_history and self.events:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} departs from its history; it learns from no event"
            )
        if (
            self.features.reads_channel_history
            and self.distance is not ScoreDistance.NEAREST_NEIGHBOURS
        ):
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} departs from its history; "
                "it has no distance to choose"
            )
        if self.events < 0:
            raise InvalidDetectorRecipeError(
                f"a detector is taught a count of events, not {self.events}"
            )
        if self.events and self.features.asks_the_pretext:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} scores by an error; it learns from no event"
            )
        if self.events and self.distance is not ScoreDistance.NEAREST_NEIGHBOURS:
            raise InvalidDetectorRecipeError(
                "a detector taught events scores by its head; it has no distance to choose"
            )
        if not self.events and self.head is not TaughtHead.DIRECTION:
            raise InvalidDetectorRecipeError(
                f"a detector taught no event has no head to choose, got {self.head}"
            )
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
        if self.normalisation is not ErrorNormalisation.RAW and not self.features.asks_the_pretext:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} scores by a distance; it has no errors to read "
                f"{self.normalisation}"
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
            not self.features.asks_the_pretext
            and not self.features.reads_channel_history
            and not self.events
            and self.distance is ScoreDistance.NEAREST_NEIGHBOURS
        )

    @property
    def is_taught(self) -> bool:
        """Whether the detector learns from a counted number of the fitting side's events."""
        return self.events > 0

    def nearest(self) -> int:
        """How many nearest fitted windows a distance is the mean to.

        Raises:
            InvalidDetectorRecipeError: If the detector scores by an error or by a distance that
                reads no neighbours.
        """
        if self.neighbours is None:
            if self.features.asks_the_pretext:
                reason = f"a detector over {self.features} scores by an error, not a distance"
            elif self.features.reads_channel_history:
                reason = f"a detector over {self.features} departs from its history, not a distance"
            elif self.events:
                reason = f"a detector taught {self.events} events reads no neighbours"
            else:
                reason = f"a detector by the {self.distance} distance reads no neighbours"
            raise InvalidDetectorRecipeError(reason)
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

    def read_history(self) -> ChannelHistory:
        """How far back each channel is read and how its departures make the score.

        Raises:
            InvalidDetectorRecipeError: If the detector does not read its channels' history.
        """
        if self.history is None:
            raise InvalidDetectorRecipeError(
                f"a detector over {self.features} reads no channel's history"
            )
        return self.history
