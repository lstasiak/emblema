from dataclasses import dataclass

from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


@dataclass(frozen=True, kw_only=True)
class PatchPlan:
    """How one patch model is made for a task: its shape, how long it learns, and its seed.

    The counterpart of an adaptation plan for a network that starts from no weights and has no
    backbone to say anything about. The schedule is the one the adapted arms learn under,
    because a patch model shares their compute budget: a comparison of two networks that were
    given different amounts of arithmetic would measure the difference in arithmetic.

    Attributes:
        spec: How a window is read and how large the model is.
        schedule: How long the task is learnt, in how large a step, under what decay.
        seed: Seed of everything the run draws: the weights, dropout, the order of windows.
    """

    spec: PatchModelSpec
    schedule: AdaptationSchedule
    seed: int

    def parameters(self) -> dict[str, str | int | float]:
        """The plan flattened to scalars, in a fixed order, for whoever records a run.

        The seed is named ``run_seed``, as an adaptation plan names it, so a row that also
        carries the seed of the draw reads the same whichever network produced it.
        """
        return {
            **self.spec.parameters(),
            "run_seed": self.seed,
            "epochs": self.schedule.epochs,
            "min_steps": self.schedule.min_steps,
            "batch_size": self.schedule.batch_size,
            "learning_rate": float(self.schedule.learning_rate),
            "weight_decay": float(self.schedule.weight_decay),
            "warmup_fraction": float(self.schedule.warmup_fraction),
            "final_lr_fraction": float(self.schedule.final_lr_fraction),
        }
