from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


class KnownArms:
    """The four ways of using a backbone, each under the name a campaign competes it by.

    What each arm does with the weights is fixed — that is what the name means — while the
    weights themselves, the strength of the low-rank update and the schedule are a campaign's
    choice, so all three come in when the process is told what it serves. A register rather than
    a table read from configuration, because the names are what a stored campaign refers to: an
    arm renamed in an environment file would leave a finished grid naming candidates nothing
    supplies.
    """

    FROM_SCRATCH = CandidateRef("from_scratch")
    FROZEN_PROBE = CandidateRef("frozen_probe")
    LORA = CandidateRef("lora")
    FULL_FINE_TUNING = CandidateRef("full_fine_tuning")

    @classmethod
    def over(
        cls, backbone: ArtifactRef, lora: LoraSpec, schedule: AdaptationSchedule
    ) -> tuple[BackboneArm, ...]:
        """Every arm over ``backbone`` under ``schedule``, in reporting order.

        The control arm draws its weights anew but has the backbone's shape, so it names the
        backbone as its architecture and nothing as its weights.
        """
        return (
            BackboneArm(
                ref=cls.FROM_SCRATCH,
                mode=TransferMode.FROM_SCRATCH,
                architecture=backbone,
                backbone=None,
                lora=None,
                schedule=schedule,
            ),
            BackboneArm(
                ref=cls.FROZEN_PROBE,
                mode=TransferMode.FROZEN_PROBE,
                architecture=backbone,
                backbone=backbone,
                lora=None,
                schedule=schedule,
            ),
            BackboneArm(
                ref=cls.LORA,
                mode=TransferMode.LORA,
                architecture=backbone,
                backbone=backbone,
                lora=lora,
                schedule=schedule,
            ),
            BackboneArm(
                ref=cls.FULL_FINE_TUNING,
                mode=TransferMode.FULL_FINE_TUNING,
                architecture=backbone,
                backbone=backbone,
                lora=None,
                schedule=schedule,
            ),
        )

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the arms are called, in reporting order — the control arm first."""
        return (cls.FROM_SCRATCH, cls.FROZEN_PROBE, cls.LORA, cls.FULL_FINE_TUNING)

    @classmethod
    def catalogue(
        cls, backbone: ArtifactRef, lora: LoraSpec, schedule: AdaptationSchedule
    ) -> BackboneArmCatalogue:
        """What the arms of ``backbone`` are, with nothing that could run one.

        What a campaign is declared against and what the process running it describes its cells
        by are then the same object, built the same way.
        """
        return BackboneArmCatalogue(cls.over(backbone, lora, schedule))
