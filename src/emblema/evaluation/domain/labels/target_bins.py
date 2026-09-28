from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidTargetBinsError
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.shared.kernel.ordering import seeded_rank


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

    def draw(self, pool: Sequence[LabelledWindow], wanted: int, seed: int) -> tuple[int, ...]:
        """Positions of ``wanted`` windows in ``pool``, spread evenly over the strata.

        Each stratum is ordered by the rank its members' keys take under the seed, and the draw
        goes round the strata taking one at a time, so the budget spreads as evenly as the strata
        sizes allow and a stratum that runs out is simply skipped. A smaller budget is a prefix
        of a larger one under the same seed, so the budgets of a curve are nested.

        Raises:
            InvalidTargetBinsError: If the pool holds fewer windows than there are bins.
        """
        strata = [
            sorted(
                group,
                key=lambda index: seeded_rank(
                    seed, pool[index].window.unit, pool[index].window.position
                ),
            )
            for group in self.of_pool(pool)
        ]
        rotation = sorted(range(len(strata)), key=lambda index: seeded_rank(seed, "stratum", index))
        taken: list[int] = []
        depth = 0
        while len(taken) < wanted:
            for stratum in rotation:
                if depth < len(strata[stratum]):
                    taken.append(strata[stratum][depth])
                    if len(taken) == wanted:
                        break
            depth += 1
        return tuple(taken)
