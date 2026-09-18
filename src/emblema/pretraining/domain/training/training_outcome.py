from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class TrainingOutcome:
    """A finished run: the weights it produced and the epochs behind them.

    A resumed run reports the epochs it ran, not the ones it inherited, so the epochs need not
    begin at zero; they do have to be consecutive, because a gap would mean an epoch was lost
    rather than trained.

    Invariants: at least one epoch; epochs consecutive and ascending; the backbone is stated by
    the last epoch and by no other, and it is the run's.

    Attributes:
        backbone: The weights kept when the objective was thrown away: the best epoch's.
        epochs: What each epoch of this run measured, in order.
    """

    backbone: ArtifactRef
    epochs: tuple[EpochOutcome, ...]

    def __post_init__(self) -> None:
        if not self.epochs:
            raise InvalidTrainingOutcomeError("a run must have trained an epoch")
        numbers = [outcome.epoch for outcome in self.epochs]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise InvalidTrainingOutcomeError(f"epochs must be consecutive, got {numbers}")
        stated = [outcome.epoch for outcome in self.epochs if outcome.backbone is not None]
        if stated != [numbers[-1]]:
            raise InvalidTrainingOutcomeError(
                f"the backbone belongs to the last epoch alone, got it on {stated}"
            )
        if self.epochs[-1].backbone != self.backbone:
            raise InvalidTrainingOutcomeError(
                "the run's backbone and its last epoch's must be the same artifact"
            )

    @property
    def best_epoch(self) -> int | None:
        """The epoch of this run whose weights the run kept; ``None`` where it kept inherited ones.

        A run picked up from a checkpoint may end with weights an epoch before the checkpoint
        wrote, which no epoch of this run reports.
        """
        kept = [epoch.epoch for epoch in self.epochs if epoch.weights == self.backbone]
        return kept[-1] if kept else None

    @property
    def hidden_ratio(self) -> float:
        """Share of observed validation tokens the masks hid, as the last epoch counted it.

        The masks a run is scored under are fixed by its seed, so every epoch hides the same
        tokens and the last one speaks for the run.
        """
        return self.epochs[-1].hidden_ratio
