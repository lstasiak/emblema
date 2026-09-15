from dataclasses import dataclass
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class EpochOutcome:
    """What one epoch measured and what it left in the artifact store.

    Two references rather than one, because the two artifacts are kept for different reasons and
    for different lengths of time: a checkpoint carries the whole state of the run and exists so
    that a dropped session can be picked up, while the backbone carries the weights alone and is
    what the run was for. The backbone appears on the last epoch of a run and nowhere else, which
    is what lets a caller consuming the epochs one by one know when it has them all.

    Invariants: the epoch is not negative; the losses are finite and not negative, being mean
    squared errors; the time taken is not negative; the share hidden lies in ``(0, 1]``, since an
    epoch that hid nothing scored nothing.

    Attributes:
        epoch: Which epoch this was, counted from zero.
        training_loss: Mean reconstruction error over the epoch's hidden tokens.
        validation_loss: The same error on the validation windows, under masks the seed fixes.
        hidden_ratio: Share of the validation windows' observed tokens those masks hid, counted
            rather than taken from what the strategy expects.
        seconds: Wall-clock time the epoch took, training and scoring together.
        checkpoint: The last resumable state written during the epoch; ``None`` where the policy
            asked for none.
        backbone: The weights kept, on the run's final epoch only.
    """

    epoch: int
    training_loss: float
    validation_loss: float
    hidden_ratio: float
    seconds: float
    checkpoint: ArtifactRef | None = None
    backbone: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise InvalidTrainingOutcomeError(f"epoch must not be negative, got {self.epoch}")
        for label, value in (
            ("training_loss", self.training_loss),
            ("validation_loss", self.validation_loss),
            ("seconds", self.seconds),
        ):
            if not isfinite(value) or value < 0.0:
                raise InvalidTrainingOutcomeError(
                    f"{label} must be finite and not negative, got {value}"
                )
        if not 0.0 < self.hidden_ratio <= 1.0:
            raise InvalidTrainingOutcomeError(
                f"hidden_ratio must lie in (0, 1], got {self.hidden_ratio}"
            )
