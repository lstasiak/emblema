from dataclasses import dataclass, field, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import InvalidBackboneArmError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.training_regime import TrainingRegime
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class BackboneArm:
    """One way of making a candidate out of a pretrained backbone, under a name of its own.

    A way of using pretrained weights is a competitor in a campaign rather than a dimension
    beside the competitors (ADR-0035), so each one is named here and enters a grid on the terms
    every other candidate enters it on. The schedule, the pooling and the encoder's setting are
    the arm's, because a variant of an arm is the arm with one of them turned and nothing else;
    the arms of one campaign still spend one budget, which the design checks off the schedules
    rather than trusting.

    Invariants: an arm whose mode solves its head in closed form names penalties, so an arm a
    campaign is declared against is one its plan can be made from. An arm that trains its head
    may hold the campaign's penalties too, for a variant that solves the head before the first
    step; they reach its plan only then.

    Attributes:
        ref: What the campaign calls this arm.
        mode: What the backbone's weights do while the task is learnt.
        architecture: Artifact of the model whose shape the arm has, whether it starts from its
            weights or draws them anew; what pins the control arm's size to the campaign.
        backbone: Artifact of the pretrained weights; ``None`` for an arm that starts from its
            architecture's initialisation.
        lora: The low-rank updates, where the mode adds them; ``None`` otherwise.
        schedule: How long and how fast the arm learns the task.
        pooling: How the states of a window become the one state the arm's head reads.
        ridge: The penalties a head solved in closed form chooses among; ``None`` where the arm
            never solves one.
        encoder: What the encoder drops while it learns and which readings it is given.
        regime: How the arm's run is stopped, weighted and perturbed inside its schedule.
    """

    ref: CandidateRef
    mode: TransferMode
    architecture: ArtifactRef
    backbone: ArtifactRef | None
    lora: LoraSpec | None
    schedule: AdaptationSchedule
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)
    ridge: RidgePenalties | None = None
    encoder: EncoderSetting = field(default_factory=EncoderSetting.standard)
    regime: TrainingRegime = field(default_factory=TrainingRegime.standard)

    def __post_init__(self) -> None:
        if self.ridge is None and self.mode.solves_the_head_in_closed_form:
            raise InvalidBackboneArmError(
                f"{self.ref} under {self.mode} solves its head in closed form and names no "
                "penalties"
            )

    @property
    def solves_a_head(self) -> bool:
        """Whether a head is solved in closed form: as the mode's answer or as a run's start."""
        return self.mode.solves_the_head_in_closed_form or self.regime.solves_the_head_first

    @property
    def solved_under(self) -> RidgePenalties | None:
        """The penalties a solved head chooses among; ``None`` where no head is solved."""
        return self.ridge if self.solves_a_head else None

    def tuned(self, knob: str, value: str) -> Self:
        """This arm with ``knob`` turned to ``value``, on the pooling, the encoder or the schedule.

        Raises:
            UnknownKnobError: If none has such a knob, or it cannot take that value.
        """
        if knob in HeadPooling.KNOBS:
            return replace(self, pooling=self.pooling.tuned(knob, value))
        if knob in EncoderSetting.KNOBS:
            return replace(self, encoder=self.encoder.tuned(knob, value))
        if knob in TrainingRegime.KNOBS:
            return replace(self, regime=self.regime.tuned(knob, value))
        return replace(self, schedule=self.schedule.tuned(knob, value))
