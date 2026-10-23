from dataclasses import dataclass
from typing import Self

from emblema.evaluation.domain.exceptions import InvalidOrderLaneError


@dataclass(frozen=True, kw_only=True)
class OrderLane:
    """One of several processes that share the cells of one order, by position.

    A cell of a campaign takes a fraction of a platform's accelerator, so one process per order
    leaves most of a paid card idle. Several processes may run one order side by side when each
    takes the cells whose position in the order's grid falls to it, every ``count``-th from
    ``index`` on: the lanes answer disjoint cells that together make the order, each reports its
    own chain of results, and every result is accepted as a result of the order is, cell by
    cell. Written ``index/count`` on a command line, the index counted from zero.

    Invariants: the count is positive and the index lies in ``[0, count)``.

    Attributes:
        index: Which lane this is, from zero.
        count: How many lanes share the order.
    """

    index: int
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise InvalidOrderLaneError(
                f"an order is shared by at least one lane, got {self.count}"
            )
        if not 0 <= self.index < self.count:
            raise InvalidOrderLaneError(
                f"a lane's index lies in [0, {self.count}), got {self.index}"
            )

    @classmethod
    def parse(cls, text: str) -> Self:
        """The lane written ``index/count``.

        Raises:
            InvalidOrderLaneError: If the text is not two whole numbers around a slash, or
                they make no lane.
        """
        index, slash, count = text.strip().partition("/")
        try:
            if not slash:
                raise ValueError("no slash")
            return cls(index=int(index), count=int(count))
        except ValueError as error:
            raise InvalidOrderLaneError(f"a lane is written index/count, got {text!r}") from error

    def takes(self, position: int) -> bool:
        """Whether the cell at ``position`` in the order's grid falls to this lane."""
        return position % self.count == self.index

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"
