from emblema.evaluation.adapters.candidates.backbone_arm import BackboneArm
from emblema.evaluation.adapters.candidates.backbone_arm_catalogue import BackboneArmCatalogue
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


class KnownArms:
    """The ways of using a backbone, each under the name a campaign competes it by.

    What each arm does with the weights is fixed — that is what the name means — while the
    weights themselves, the strength of the low-rank update, the penalties a head solved in
    closed form chooses among and the schedule are a campaign's choice, so all four come in when
    the process is told what it serves. Every arm that trains its head holds the penalties too,
    for a variant that solves the head before its first step. The probe over the encoder at its
    initialisation is an arm of its own name, not a backbone left out, so a campaign never reads
    one by mistake for the other. A register rather than
    a table read from configuration, because the names are what a stored campaign refers to: an
    arm renamed in an environment file would leave a finished grid naming candidates nothing
    supplies.
    """

    FROM_SCRATCH = CandidateRef("from_scratch")
    FROZEN_PROBE = CandidateRef("frozen_probe")
    FROZEN_RIDGE = CandidateRef("frozen_ridge")
    LORA = CandidateRef("lora")
    FULL_FINE_TUNING = CandidateRef("full_fine_tuning")
    UNTRAINED_RIDGE = CandidateRef("untrained_ridge")

    @classmethod
    def over(
        cls,
        backbone: ArtifactRef,
        lora: LoraSpec,
        schedule: AdaptationSchedule,
        ridge: RidgePenalties,
    ) -> tuple[BackboneArm, ...]:
        """Every arm over ``backbone`` under ``schedule``, in reporting order.

        The control arm and the probe over the encoder at its initialisation draw their weights
        anew but have the backbone's shape, so they name the backbone as their architecture and
        nothing as their weights.
        """
        return (
            BackboneArm(
                ref=cls.FROM_SCRATCH,
                mode=TransferMode.FROM_SCRATCH,
                architecture=backbone,
                backbone=None,
                lora=None,
                schedule=schedule,
                ridge=ridge,
            ),
            BackboneArm(
                ref=cls.FROZEN_PROBE,
                mode=TransferMode.FROZEN_PROBE,
                architecture=backbone,
                backbone=backbone,
                lora=None,
                schedule=schedule,
                ridge=ridge,
            ),
            BackboneArm(
                ref=cls.FROZEN_RIDGE,
                mode=TransferMode.FROZEN_RIDGE,
                architecture=backbone,
                backbone=backbone,
                lora=None,
                schedule=schedule,
                ridge=ridge,
            ),
            BackboneArm(
                ref=cls.LORA,
                mode=TransferMode.LORA,
                architecture=backbone,
                backbone=backbone,
                lora=lora,
                schedule=schedule,
                ridge=ridge,
            ),
            BackboneArm(
                ref=cls.FULL_FINE_TUNING,
                mode=TransferMode.FULL_FINE_TUNING,
                architecture=backbone,
                backbone=backbone,
                lora=None,
                schedule=schedule,
                ridge=ridge,
            ),
            BackboneArm(
                ref=cls.UNTRAINED_RIDGE,
                mode=TransferMode.FROZEN_RIDGE,
                architecture=backbone,
                backbone=None,
                lora=None,
                schedule=schedule,
                ridge=ridge,
            ),
        )

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the arms are called, in reporting order — the control arm first."""
        return (
            cls.FROM_SCRATCH,
            cls.FROZEN_PROBE,
            cls.FROZEN_RIDGE,
            cls.LORA,
            cls.FULL_FINE_TUNING,
            cls.UNTRAINED_RIDGE,
        )

    @classmethod
    def catalogue(
        cls,
        backbone: ArtifactRef,
        lora: LoraSpec,
        schedule: AdaptationSchedule,
        ridge: RidgePenalties,
    ) -> BackboneArmCatalogue:
        """What the arms of ``backbone`` are, with nothing that could run one.

        What a campaign is declared against and what the process running it describes its cells
        by are then the same object, built the same way.
        """
        return BackboneArmCatalogue(cls.over(backbone, lora, schedule, ridge))
