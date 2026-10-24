from dataclasses import dataclass
from math import isfinite
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidObservationTermError


@dataclass(frozen=True, kw_only=True)
class ObservationTerm:
    """The terms of a pretext over cells that ask about the measuring, not the measured values.

    A hidden cell is one channel over one span of the window. Beside the loss on what the span
    held, a run may ask whether anything was measured there at all, at every hidden cell, and
    how often, where something was: the process of observation, which on the intensive-care
    stays carries the task nearly as far as the values do. Each is weighed against one hidden
    cell's loss on its values. The term is a parameter of the method like the loss, stated by the
    experiment, reported with the run and digested into its signature; a run that states none
    renders nothing, as a run over cells that predicts their values alone.

    Invariants: both weights are finite and not negative; both at zero is no term.

    Attributes:
        presence_weight: What the presence of one hidden cell weighs against its values' loss.
        count_weight: What the count of one hidden cell that held a reading weighs against it.
    """

    presence_weight: float = 0.0
    count_weight: float = 0.0

    def __post_init__(self) -> None:
        for label, weight in (
            ("presence_weight", self.presence_weight),
            ("count_weight", self.count_weight),
        ):
            if not isfinite(weight) or weight < 0.0:
                raise InvalidObservationTermError(
                    f"{label} must be finite and not negative, got {weight}"
                )

    @classmethod
    def none(cls) -> Self:
        """No term: the cells' values alone."""
        return cls()

    @classmethod
    def of(cls, *, presence_weight: float = 0.0, count_weight: float = 0.0) -> Self:
        return cls(presence_weight=presence_weight, count_weight=count_weight)

    @property
    def is_stated(self) -> bool:
        return self.presence_weight > 0.0 or self.count_weight > 0.0

    @property
    def asks_presence(self) -> bool:
        return self.presence_weight > 0.0

    @property
    def asks_count(self) -> bool:
        return self.count_weight > 0.0
