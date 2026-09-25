from dataclasses import dataclass, field

from emblema.evaluation.domain.exceptions import InvalidAdaptationPlanError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class AdaptationPlan:
    """How one candidate is made out of the backbone for a task: mode, weights, schedule, seed.

    One value holds everything a cell of the curve does to the backbone, so two cells can be told
    apart by comparing plans and a run reports the plan it was made under. Which weights it
    starts from is part of the plan rather than of the mode: the control arm starts from none,
    and every other mode from the artifact of a pretrained backbone, held by reference and
    checksum as the registry holds it.

    Invariants: weights are named exactly when the mode starts from pretrained ones; low-rank
    updates are specified exactly when the mode adds them.

    Attributes:
        mode: What the backbone's weights do while the task is learnt.
        backbone: Artifact of the pretrained weights the run starts from; ``None`` for the
            control arm.
        schedule: How long the task is learnt, in how large a step, under what decay.
        lora: The low-rank updates, where the mode adds them; ``None`` otherwise.
        seed: Seed of everything the run draws: the head, fresh backbone weights, the low-rank
            updates, the order of windows. Apart from the seed of the draw and outside the
            schedule: another seed is a repeat of the same schedule over the same labels.
        pooling: How the states of a window become the one state the head reads; the mean
            over the window unless a variant turns it.
    """

    mode: TransferMode
    backbone: ArtifactRef | None
    schedule: AdaptationSchedule
    lora: LoraSpec | None
    seed: int
    pooling: HeadPooling = field(default_factory=HeadPooling.mean)

    def __post_init__(self) -> None:
        if (self.backbone is None) == self.mode.starts_from_pretrained_weights:
            raise InvalidAdaptationPlanError(
                f"{self.mode} "
                + (
                    "starts from pretrained weights and names none"
                    if self.backbone is None
                    else "starts from no weights and names some"
                )
            )
        if (self.lora is None) == self.mode.adds_low_rank_updates:
            raise InvalidAdaptationPlanError(
                f"{self.mode} "
                + (
                    "adds low-rank updates and specifies none"
                    if self.lora is None
                    else "adds no low-rank updates and specifies some"
                )
            )

    def parameters(self) -> dict[str, str | int | float]:
        """The plan flattened to scalars, in a fixed order, for whoever reports a run.

        Absent parts are rendered as empty text and zeros rather than left out, so every plan
        renders the same columns and two plans can be compared by comparing this mapping. The
        seed is named ``run_seed``, so a row that also carries the seed of the draw reads
        unambiguously.
        """
        return {
            "mode": str(self.mode),
            "backbone": "" if self.backbone is None else self.backbone.key,
            "run_seed": self.seed,
            "epochs": self.schedule.epochs,
            "min_steps": self.schedule.min_steps,
            "batch_size": self.schedule.batch_size,
            "learning_rate": float(self.schedule.learning_rate),
            "weight_decay": float(self.schedule.weight_decay),
            "warmup_fraction": float(self.schedule.warmup_fraction),
            "final_lr_fraction": float(self.schedule.final_lr_fraction),
            **self.pooling.parameters(),
            "lora_rank": 0 if self.lora is None else self.lora.rank,
            "lora_alpha": 0.0 if self.lora is None else float(self.lora.alpha),
            "lora_dropout": 0.0 if self.lora is None else float(self.lora.dropout),
            "lora_targets": "" if self.lora is None else " ".join(self.lora.targets),
        }
