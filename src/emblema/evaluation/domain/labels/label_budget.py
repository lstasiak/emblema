from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidLabelBudgetError


@dataclass(frozen=True)
class LabelBudget:
    """How many labelled windows a run is allowed to learn from.

    The unit is a labelled window, not a labelled unit and not an observation: an engine yields
    tens of windows, so counting units would measure something far coarser than the axis of the
    curve, and a label is attached to a window rather than to each of its observations. The
    largest budget is everything the tuning side holds, which is a number nobody knows when the
    budget is stated, so it is its own case rather than a figure written down.

    Invariants: a counted budget asks for at least one window.

    Attributes:
        windows: How many windows to draw, or ``None`` for every window available.
    """

    windows: int | None

    def __post_init__(self) -> None:
        if self.windows is not None and self.windows < 1:
            raise InvalidLabelBudgetError(
                f"a budget must ask for at least one window: {self.windows}"
            )

    @classmethod
    def of(cls, windows: int) -> Self:
        """A budget of exactly this many windows."""
        return cls(windows)

    @classmethod
    def everything(cls) -> Self:
        """A budget of every labelled window the tuning side holds."""
        return cls(None)

    def drawn_from(self, pool: int) -> int:
        """How many windows this budget takes from a pool of ``pool`` windows.

        Raises:
            InvalidLabelBudgetError: If the pool holds fewer windows than the budget asks for.
        """
        if self.windows is None:
            return pool
        if self.windows > pool:
            raise InvalidLabelBudgetError(
                f"budget of {self.windows} windows exceeds the {pool} the task holds"
            )
        return self.windows
