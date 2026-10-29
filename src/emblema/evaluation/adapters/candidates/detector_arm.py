from dataclasses import dataclass, field, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
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
    readings — and the knobs of the detector itself: the neighbours a score is the distance to,
    the share of the fitting side the alarm flags and what a distance is read against. A variant
    of an arm is the arm with one of those turned, or the encoder's layer or pooling turned.

    Invariants: an arm over an encoder's states names the model whose shape it reads; one over
    the summaries names none.

    Attributes:
        ref: What the campaign calls this detector.
        features: What a window is read as.
        architecture: Artifact of the model whose shape an encoder has; ``None`` for the
            summaries.
        backbone: Artifact of the pretrained weights; ``None`` for the shape at its
            initialisation, and for the summaries.
        encoder: Which readings the encoder is given and which layer it is read at.
        pooling: How the states of a window become the one state the detector reads.
        neighbours: How many nearest fitted windows a score is the mean distance to.
        contamination: The share of fitting windows the alarm is set to flag.
        reference: What a window's distance is read against before the alarm.
    """

    KNOBS = ("neighbours", "contamination", "reference")

    ref: CandidateRef
    features: DetectorFeatures
    architecture: ArtifactRef | None
    backbone: ArtifactRef | None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)
    neighbours: int = 5
    contamination: float = 0.02
    reference: ScoreReference = ScoreReference.FITTING_SIDE

    def __post_init__(self) -> None:
        if (self.architecture is None) == self.features.reads_an_encoder:
            raise InvalidDetectorRecipeError(
                f"{self.ref} over {self.features} "
                + ("names no encoder" if self.architecture is None else "names an encoder")
            )
        if self.backbone is not None and self.architecture is None:
            raise InvalidDetectorRecipeError(f"{self.ref} names weights and no shape for them")
        # The recipe's own checks, run here so an arm a campaign is declared against is one a
        # cell can be made from.
        self.recipe(seed=0)

    def tuned(self, knob: str, value: str) -> Self:
        """This arm with ``knob`` turned to ``value``, on the detector, the pooling or the encoder.

        Raises:
            UnknownKnobError: If none has such a knob, or it cannot take that value.
        """
        if knob == "neighbours":
            return replace(self, neighbours=self._count(knob, value))
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
            raise UnknownKnobError(
                f"a detector over {self.features} turns {self.KNOBS} and nothing else"
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
            neighbours=self.neighbours,
            contamination=self.contamination,
            seed=seed,
            reference=self.reference,
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
