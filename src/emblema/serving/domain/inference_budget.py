from collections.abc import Sequence
from dataclasses import dataclass

from emblema.serving.domain.exceptions import InvalidInferenceBudgetError, WindowBeyondBudgetError


@dataclass(frozen=True, kw_only=True)
class InferenceBudget:
    """How much a process may run through its networks at once, and how a request is cut to fit.

    Attention compares every token of a window with every other, so what a batch holds in
    memory grows with the square of its longest window and with the number of windows padded to
    it. That product, in token pairs, is the cost this counts; it is a proxy for memory and not
    a measurement of it, which is what a limit on concurrency needs: the same ordering of what
    is heavier, in a unit an operator can reason about.

    The budget is stated as how many windows of the longest length the service admits may run
    at once. A batch never costs more than one such window, so a request of long windows is cut
    into small batches and a request of short ones into large, and either fits beside another
    request rather than holding the whole budget while it runs.

    Invariants: both counts are positive.

    Attributes:
        windows: How many windows of the longest admitted length may run at once.
        longest: The most tokens a window may hold; the longest the service admits.
    """

    windows: int
    longest: int

    def __post_init__(self) -> None:
        for label, count in (("windows", self.windows), ("longest", self.longest)):
            if count < 1:
                raise InvalidInferenceBudgetError(f"{label} must be positive, got {count}")

    @property
    def capacity(self) -> int:
        """Token pairs that may be running at once, across every request."""
        return self.windows * self.ceiling

    @property
    def ceiling(self) -> int:
        """Token pairs one batch may cost: one window of the longest admitted length."""
        return self.longest * self.longest

    @staticmethod
    def cost(count: int, longest: int) -> int:
        """Token pairs of ``count`` windows padded to ``longest`` tokens."""
        return count * longest * longest

    def plan(self, lengths: Sequence[int], *, max_windows: int) -> tuple[tuple[int, ...], ...]:
        """How to cut windows of these lengths into batches that each fit under the ceiling.

        The longest go first and share batches with each other, so that no short window is
        padded to a long one. A batch holds at most ``max_windows`` windows.

        Args:
            lengths: Tokens of each window, in the order the caller holds them.
            max_windows: The most windows one batch may hold.

        Returns:
            The positions in ``lengths`` that make each batch, longest windows first.

        Raises:
            WindowBeyondBudgetError: If one window alone costs more than a batch may.
        """
        order = sorted(range(len(lengths)), key=lambda position: -lengths[position])
        batches: list[list[int]] = []
        for position in order:
            if self.cost(1, lengths[position]) > self.ceiling:
                raise WindowBeyondBudgetError(
                    f"a window of {lengths[position]} tokens costs more than the budget lets a "
                    f"batch cost: at most {self.longest} tokens"
                )
            if (
                batches
                and len(batches[-1]) < max_windows
                and self.cost(len(batches[-1]) + 1, lengths[batches[-1][0]]) <= self.ceiling
            ):
                batches[-1].append(position)
            else:
                batches.append([position])
        return tuple(tuple(batch) for batch in batches)
