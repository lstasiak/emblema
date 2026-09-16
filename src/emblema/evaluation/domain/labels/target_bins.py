from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidTargetBinsError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow


@dataclass(frozen=True)
class TargetBins:
    """The strata a budget is drawn across: windows ordered by target, cut into equal groups.

    A small budget drawn without strata is a lottery over the range of the target — a draw of
    fifty windows can miss failure entirely and measure a model on a question it was never
    asked. Groups are cut by rank rather than by value because a quarter of the targets sit
    exactly at the scheme's ceiling: value thresholds would put that tie in one bin and leave
    others empty, while ranks keep every group within one window of the same size.

    Invariants: at least one bin.

    Attributes:
        count: How many strata the pool is cut into.
    """

    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise InvalidTargetBinsError(f"a stratification needs at least one bin: {self.count}")

    def of_pool(self, pool: Sequence[LabelledWindow]) -> tuple[tuple[int, ...], ...]:
        """Positions in ``pool``, grouped into strata, each group in order of increasing target.

        Ties are broken by the window's unit and position, so the same pool always cuts the same
        way whatever order it arrived in.

        Raises:
            InvalidTargetBinsError: If the pool holds fewer windows than there are bins.
        """
        if len(pool) < self.count:
            raise InvalidTargetBinsError(f"{len(pool)} windows cannot fill {self.count} bins")
        ranked = sorted(
            range(len(pool)),
            key=lambda index: (
                pool[index].target,
                str(pool[index].window.unit),
                pool[index].window.position,
            ),
        )
        size, remainder = divmod(len(ranked), self.count)
        groups = []
        start = 0
        for bin_index in range(self.count):
            end = start + size + (1 if bin_index < remainder else 0)
            groups.append(tuple(ranked[start:end]))
            start = end
        return tuple(groups)
