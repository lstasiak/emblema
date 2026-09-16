from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.shared.kernel.ordering import seeded_rank


@dataclass(frozen=True, kw_only=True)
class LabelSample:
    """The labelled windows one run is allowed to see, and what drew them.

    A point on a label-efficiency curve is only a point if it can be drawn again: the task, the
    seed and the budget are carried with the windows so that the same run repeats and two methods
    compared at one budget are compared on the same labels. Two samples of equal size say nothing
    about each other unless they name the task they came out of.

    Attributes:
        task: Task the windows were drawn from.
        windows: The drawn windows, in a fixed order.
        budget: What was asked for.
        seed: Seed the draw was made under.
    """

    task: TaskId
    windows: tuple[LabelledWindow, ...]
    budget: LabelBudget
    seed: int

    @classmethod
    def drawn(
        cls,
        task: TaskId,
        pool: Sequence[LabelledWindow],
        budget: LabelBudget,
        bins: TargetBins,
        seed: int,
    ) -> Self:
        """Draw ``budget`` windows out of ``pool``, spread evenly over the strata of the target.

        Each stratum is ordered by the rank its members' keys take under the seed, and the draw
        goes round the strata taking one at a time, so the budget spreads as evenly as the strata
        sizes allow and a stratum that runs out is simply skipped. Ranking each window on its own
        rather than shuffling the pool is what makes the draw independent of the order the windows
        arrived in, so it repeats on another machine and after a corpus grows.

        A budget that takes the whole pool is not drawn at all: there is nothing to spread, and a
        task too small to fill its strata would otherwise be refused the one budget it can serve.

        Raises:
            InvalidLabelBudgetError: If the pool holds fewer windows than the budget asks for.
            InvalidTargetBinsError: If a drawn budget leaves fewer windows than there are bins.
        """
        wanted = budget.drawn_from(len(pool))
        if wanted == len(pool):
            whole = cls._ordered(pool, range(len(pool)))
            return cls(task=task, windows=whole, budget=budget, seed=seed)
        strata = [
            sorted(
                group,
                key=lambda index: seeded_rank(
                    seed, pool[index].window.unit, pool[index].window.position
                ),
            )
            for group in bins.of_pool(pool)
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
        return cls(task=task, windows=cls._ordered(pool, taken), budget=budget, seed=seed)

    @staticmethod
    def _ordered(
        pool: Sequence[LabelledWindow], taken: Iterable[int]
    ) -> tuple[LabelledWindow, ...]:
        """The drawn windows by unit and place, so a sample reads the same however it was drawn."""
        ordered = sorted(
            taken, key=lambda index: (str(pool[index].window.unit), pool[index].window.position)
        )
        return tuple(pool[index] for index in ordered)
