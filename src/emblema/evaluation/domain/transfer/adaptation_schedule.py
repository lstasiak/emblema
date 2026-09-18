from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidAdaptationScheduleError


@dataclass(frozen=True, kw_only=True)
class AdaptationSchedule:
    """How long a task is learnt from its labels, in how large a step, and under what decay.

    A fixed number of epochs, scored after the last: stopping when the validation error turns
    would spend the validation side on a decision per run, and every number reported on it
    afterwards would be a little too good. The weight decay is stated rather than inherited,
    because a mode that caps the degrees of freedom is compared against modes that regularise
    otherwise, and what those did has to be on the record.

    Invariants: the epochs and the batch size are positive; the learning rate is positive and
    finite; the weight decay is finite and not negative.

    Attributes:
        epochs: Passes over the labelled windows.
        batch_size: Windows per optimiser step.
        learning_rate: Step size of the optimiser.
        weight_decay: Decoupled weight decay of the optimiser; zero for none.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float

    def __post_init__(self) -> None:
        for label, count in (("epochs", self.epochs), ("batch_size", self.batch_size)):
            if count < 1:
                raise InvalidAdaptationScheduleError(f"{label} must be positive, got {count}")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise InvalidAdaptationScheduleError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise InvalidAdaptationScheduleError(
                f"weight_decay must be finite and not negative, got {self.weight_decay}"
            )
