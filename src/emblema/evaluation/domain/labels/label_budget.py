from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidLabelBudgetError

EVERYTHING = "all"
"""What the budget of every labelled window the tuning side holds is called."""


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

    @classmethod
    def parse(cls, text: str) -> Self:
        """The budget that canonical text names.

        Raises:
            InvalidLabelBudgetError: If the text is neither a count of windows nor the word for
                every window there is.
        """
        if text == EVERYTHING:
            return cls.everything()
        try:
            return cls.of(int(text))
        except ValueError as error:
            raise InvalidLabelBudgetError(f"not a budget: {text!r}") from error

    def text(self) -> str:
        """The budget as one word, its canonical form: a count, or every window there is.

        A budget is a coordinate — of a cell, of a row in a report, of a key in a store — and a
        coordinate needs one spelling. The largest budget has no count to write, so it is named
        rather than numbered.
        """
        return EVERYTHING if self.windows is None else str(self.windows)

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
