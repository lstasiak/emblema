from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class BackboneArm:
    """One way of making a candidate out of a pretrained backbone, under a name of its own.

    A way of using pretrained weights is a competitor in a campaign rather than a dimension
    beside the competitors (ADR-0035), so each one is named here and enters a grid on the terms
    every other candidate enters it on.

    Attributes:
        ref: What the campaign calls this arm.
        mode: What the backbone's weights do while the task is learnt.
        backbone: Artifact of the pretrained weights; ``None`` for the arm that starts from
            none.
        lora: The low-rank updates, where the mode adds them; ``None`` otherwise.
    """

    ref: CandidateRef
    mode: TransferMode
    backbone: ArtifactRef | None
    lora: LoraSpec | None
