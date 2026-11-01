from dataclasses import dataclass, field, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.hidden_cells_choice import HiddenCellsChoice
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError, UnknownKnobError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class DetectorArm:
    """One detector a campaign competes, under a name of its own.

    What separates one of these from another is what a window is read as — an encoder's
    states, pretrained or at their initialisation, the cells the encoder would read, the
    summaries per channel or their counts of readings, or the error of what is filled in under a
    backbone's masks — and the knobs of the detector itself: which distance from the fitted
    windows it scores by and the neighbours that distance is taken to, or which cells are hidden,
    the draws an error is averaged over and how the errors are read, the share of the fitting
    side the alarm flags and what a score is read against. A variant of an arm is the arm with
    one of those turned, or the encoder's layer or pooling turned.

    Invariants: an arm over an encoder's states names the model whose shape it reads; an arm
    that asks the pretext names the backbone whose masks it asks under and no shape; one over
    the summaries names neither.

    Attributes:
        ref: What the campaign calls this detector.
        features: What a window is read as.
        architecture: Artifact of the model whose shape an encoder has; ``None`` for the
            summaries.
        backbone: Artifact of the pretrained weights; ``None`` for the shape at its
            initialisation, and for the summaries.
        encoder: Which readings the encoder is given and which layer it is read at.
        pooling: How the states of a window become the one state the detector reads.
        distance: Which distance from the fitted windows a detector over features scores by.
        neighbours: How many nearest fitted windows that distance is the mean to, where it is
            the distance to the nearest.
        contamination: The share of fitting windows the alarm is set to flag.
        reference: What a window's score is read against before the alarm.
        draws: How many draws of the backbone's masks an error is averaged over.
        hidden: Which cells the pretext is asked to fill in.
        normalisation: How the cells' errors become a window's score.
        events: How many of the fitting side's events a detector over features is taught.
        head: How a taught detector scores a window by its events.
    """

    KNOBS = (
        "distance",
        "neighbours",
        "contamination",
        "reference",
        "draws",
        "hidden",
        "error",
        "events",
        "head",
    )
    DEFAULT_NEIGHBOURS = 5

    ref: CandidateRef
    features: DetectorFeatures
    architecture: ArtifactRef | None
    backbone: ArtifactRef | None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)
    distance: ScoreDistance = ScoreDistance.NEAREST_NEIGHBOURS
    neighbours: int = DEFAULT_NEIGHBOURS
    contamination: float = 0.02
    reference: ScoreReference = ScoreReference.FITTING_SIDE
    draws: int = PretextAsking.STANDARD_DRAWS
    hidden: HiddenCellsChoice = HiddenCellsChoice.OWN
    normalisation: ErrorNormalisation = ErrorNormalisation.RAW
    events: int = 0
    head: TaughtHead = TaughtHead.DIRECTION

    def __post_init__(self) -> None:
        if (self.architecture is None) == self.features.reads_an_encoder:
            raise InvalidDetectorRecipeError(
                f"{self.ref} over {self.features} "
                + ("names no encoder" if self.architecture is None else "names an encoder")
            )
        if self.features.asks_the_pretext:
            if self.backbone is None:
                raise InvalidDetectorRecipeError(f"{self.ref} names no backbone to ask")
        elif self.backbone is not None and self.architecture is None:
            raise InvalidDetectorRecipeError(f"{self.ref} names weights and no shape for them")
        # The recipe's own checks, run here so an arm a campaign is declared against is one a
        # cell can be made from.
        self.recipe(seed=0)

    def tuned(self, knob: str, value: str) -> Self:
        """This arm with ``knob`` turned to ``value``, on the detector, the pooling or the encoder.

        Only the distance to the nearest neighbours reads neighbours, a detector taught events
        reads neither a distance nor neighbours, and a hidden tail is asked once, so a name that
        turns both of such a pair is refused whichever comes first, as is a knob turned to the
        value it already has: two names would otherwise denote one detector. A head is chosen
        for a detector taught events, so the events come first in the name.

        Raises:
            UnknownKnobError: If none has such a knob, or it cannot take that value.
        """
        if knob == "distance" and not self.features.asks_the_pretext:
            try:
                distance = ScoreDistance(value)
            except ValueError as error:
                raise UnknownKnobError(
                    f"{knob} takes one of {[str(d) for d in ScoreDistance]}, not {value!r}"
                ) from error
            if self.events:
                raise UnknownKnobError("a detector taught events has no distance to choose")
            if (
                distance is not ScoreDistance.NEAREST_NEIGHBOURS
                and self.neighbours != self.DEFAULT_NEIGHBOURS
            ):
                raise UnknownKnobError(f"the {distance} distance reads no neighbours")
            return replace(self, distance=distance)
        if knob == "neighbours" and not self.features.asks_the_pretext:
            if self.events:
                raise UnknownKnobError("a detector taught events reads no neighbours")
            if self.distance is not ScoreDistance.NEAREST_NEIGHBOURS:
                raise UnknownKnobError(f"the {self.distance} distance reads no neighbours")
            return replace(self, neighbours=self._turned(knob, value, self.neighbours))
        if knob == "events" and not self.features.asks_the_pretext:
            if (
                self.distance is not ScoreDistance.NEAREST_NEIGHBOURS
                or self.neighbours != self.DEFAULT_NEIGHBOURS
            ):
                raise UnknownKnobError(
                    "a detector taught events scores by its head; it reads no distance and no "
                    "neighbours"
                )
            return replace(self, events=self._turned(knob, value, self.events))
        if knob == "head" and not self.features.asks_the_pretext:
            try:
                head = TaughtHead(value)
            except ValueError as error:
                raise UnknownKnobError(
                    f"{knob} takes one of {[str(h) for h in TaughtHead]}, not {value!r}"
                ) from error
            if not self.events:
                raise UnknownKnobError(
                    "a head is chosen for a detector taught events; name the events first"
                )
            if head is self.head:
                raise UnknownKnobError(f"{knob} is already {head}; the name turns nothing")
            return replace(self, head=head)
        if knob == "draws" and self.features.asks_the_pretext:
            if self.hidden is HiddenCellsChoice.TAIL:
                raise UnknownKnobError("a hidden tail is asked once; it takes no draws")
            return replace(self, draws=self._turned(knob, value, self.draws))
        if knob == "hidden" and self.features.asks_the_pretext:
            try:
                hidden = HiddenCellsChoice(value)
            except ValueError as error:
                raise UnknownKnobError(
                    f"{knob} takes one of {[str(h) for h in HiddenCellsChoice]}, not {value!r}"
                ) from error
            if hidden is HiddenCellsChoice.TAIL:
                if self.draws != PretextAsking.STANDARD_DRAWS:
                    raise UnknownKnobError("a hidden tail is asked once; it takes no draws")
                return replace(self, hidden=hidden, draws=1)
            return replace(self, hidden=hidden)
        if knob == "error" and self.features.asks_the_pretext:
            try:
                return replace(self, normalisation=ErrorNormalisation(value))
            except ValueError as error:
                raise UnknownKnobError(
                    f"{knob} takes one of {[str(n) for n in ErrorNormalisation]}, not {value!r}"
                ) from error
        if knob == "contamination":
            return replace(self, contamination=self._share(knob, value))
        if knob == "reference":
            try:
                return replace(self, reference=ScoreReference(value))
            except ValueError as error:
                raise UnknownKnobError(
                    f"{knob} takes one of {[str(r) for r in ScoreReference]}, not {value!r}"
                ) from error
        if not self.features.reads_an_encoder:
            turned = (
                ("draws", "hidden", "error")
                if self.features.asks_the_pretext
                else ("distance", "neighbours", "events", "head")
            )
            raise UnknownKnobError(
                f"a detector over {self.features} turns "
                f"{(*turned, 'contamination', 'reference')} and nothing else"
            )
        if knob in HeadPooling.KNOBS:
            return replace(self, pooling=self.pooling.tuned(knob, value))
        if knob in EncoderSetting.KNOBS:
            return replace(self, encoder=self.encoder.tuned(knob, value))
        raise UnknownKnobError(f"a detector has no knob {knob!r}")

    def recipe(self, *, seed: int) -> DetectorRecipe:
        """The recipe a cell under ``seed`` fits this arm by.

        Raises:
            InvalidDetectorRecipeError: If the arm's reading or its knobs do not hold together.
        """
        return DetectorRecipe(
            features=self.features,
            reading=(
                None
                if self.architecture is None
                else EncoderReading(
                    architecture=self.architecture,
                    backbone=self.backbone,
                    encoder=self.encoder,
                    pooling=self.pooling,
                )
            ),
            neighbours=self.neighbours if self._scores_by_nearest_neighbours else None,
            contamination=self.contamination,
            seed=seed,
            distance=self.distance,
            normalisation=self.normalisation,
            reference=self.reference,
            pretext=(
                PretextAsking(backbone=self.backbone, draws=self.draws, hidden=self.hidden)
                if self.features.asks_the_pretext and self.backbone is not None
                else None
            ),
            events=self.events,
            head=self.head,
        )

    @property
    def _scores_by_nearest_neighbours(self) -> bool:
        return (
            not self.features.asks_the_pretext
            and not self.events
            and self.distance is ScoreDistance.NEAREST_NEIGHBOURS
        )

    @classmethod
    def _turned(cls, knob: str, value: str, current: int) -> int:
        """The count ``knob`` is turned to, refused where it is the count it already has.

        Raises:
            UnknownKnobError: If the value is no count, or turns nothing.
        """
        count = cls._count(knob, value)
        if count == current:
            raise UnknownKnobError(f"{knob} is already {current}; the name turns nothing")
        return count

    @staticmethod
    def _count(knob: str, value: str) -> int:
        try:
            count = int(value)
        except ValueError as error:
            raise UnknownKnobError(f"{knob} takes a count, not {value!r}") from error
        if count < 1:
            raise UnknownKnobError(f"{knob} takes at least one, not {value!r}")
        return count

    @staticmethod
    def _share(knob: str, value: str) -> float:
        try:
            share = float(value)
        except ValueError as error:
            raise UnknownKnobError(f"{knob} takes a share, not {value!r}") from error
        if not 0.0 < share < 1.0:
            raise UnknownKnobError(f"{knob} takes a share strictly between zero and one")
        return share
