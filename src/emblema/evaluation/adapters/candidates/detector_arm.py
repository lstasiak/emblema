from dataclasses import dataclass, field, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError, UnknownKnobError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class DetectorArm:
    """One detector a campaign competes, under a name of its own.

    What separates one of these from another is what a window is read as — an encoder's
    states, pretrained or at their initialisation, the summaries per channel or their counts of
    readings, or the error of what is filled in under a backbone's masks — and the knobs of the
    detector itself: the neighbours a distance is taken to or the draws an error is averaged
    over, the share of the fitting side the alarm flags and what a score is read against. A
    variant of an arm is the arm with one of those turned, or the encoder's layer or pooling
    turned.

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
        neighbours: How many nearest fitted windows a distance is the mean to.
        contamination: The share of fitting windows the alarm is set to flag.
        reference: What a window's score is read against before the alarm.
        draws: How many draws of the backbone's masks an error is averaged over.
    """

    KNOBS = ("neighbours", "contamination", "reference", "draws")

    ref: CandidateRef
    features: DetectorFeatures
    architecture: ArtifactRef | None
    backbone: ArtifactRef | None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)
    neighbours: int = 5
    contamination: float = 0.02
    reference: ScoreReference = ScoreReference.FITTING_SIDE
    draws: int = PretextAsking.STANDARD_DRAWS

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

        Raises:
            UnknownKnobError: If none has such a knob, or it cannot take that value.
        """
        if knob == "neighbours" and not self.features.asks_the_pretext:
            return replace(self, neighbours=self._count(knob, value))
        if knob == "draws" and self.features.asks_the_pretext:
            return replace(self, draws=self._count(knob, value))
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
            turned = ("draws" if self.features.asks_the_pretext else "neighbours",)
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
            neighbours=None if self.features.asks_the_pretext else self.neighbours,
            contamination=self.contamination,
            seed=seed,
            reference=self.reference,
            pretext=(
                PretextAsking(backbone=self.backbone, draws=self.draws)
                if self.features.asks_the_pretext and self.backbone is not None
                else None
            ),
        )

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
