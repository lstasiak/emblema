from dataclasses import dataclass, field, replace
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class BackboneArm:
    """One way of making a candidate out of a pretrained backbone, under a name of its own.

    A way of using pretrained weights is a competitor in a campaign rather than a dimension
    beside the competitors (ADR-0035), so each one is named here and enters a grid on the terms
    every other candidate enters it on. The schedule and the pooling are the arm's, because a
    variant of an arm is the arm under a turned schedule or a turned pooling and nothing else;
    the arms of one campaign still spend one budget, which the design checks off the schedules
    rather than trusting.

    Attributes:
        ref: What the campaign calls this arm.
        mode: What the backbone's weights do while the task is learnt.
        architecture: Artifact of the model whose shape the arm has, whether it starts from its
            weights or draws them anew; what pins the control arm's size to the campaign.
        backbone: Artifact of the pretrained weights; ``None`` for the arm that starts from
            none.
        lora: The low-rank updates, where the mode adds them; ``None`` otherwise.
        schedule: How long and how fast the arm learns the task.
        pooling: How the states of a window become the one state the arm's head reads.
    """

    ref: CandidateRef
    mode: TransferMode
    architecture: ArtifactRef
    backbone: ArtifactRef | None
    lora: LoraSpec | None
    schedule: AdaptationSchedule
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)

    def tuned(self, knob: str, value: str) -> Self:
        """This arm with ``knob`` turned to ``value``, on the pooling or on the schedule.

        Raises:
            UnknownKnobError: If neither has such a knob, or it cannot take that value.
        """
        if knob in HeadPooling.KNOBS:
            return replace(self, pooling=self.pooling.tuned(knob, value))
        return replace(self, schedule=self.schedule.tuned(knob, value))
