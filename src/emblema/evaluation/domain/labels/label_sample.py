from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import EmptyLabelSampleError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.stratification import Stratification


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
        strata: Stratification,
        seed: int,
    ) -> Self:
        """Draw ``budget`` windows out of ``pool``, spread over the strata the task names.

        Every window is ranked on its own under the seed rather than the pool shuffled, which is
        what makes the draw independent of the order the windows arrived in, so it repeats on
        another machine and after a corpus grows.

        A budget that takes the whole pool is not drawn at all: there is nothing to spread, and a
        task too small to fill its strata would otherwise be refused the one budget it can serve.

        Raises:
            InvalidLabelBudgetError: If the pool holds fewer windows than the budget asks for.
            InvalidTargetBinsError: If a drawn budget leaves fewer windows than there are bins.
            SingleClassSampleError: If a drawn budget of a binary task holds one outcome only.
        """
        wanted = budget.drawn_from(len(pool))
        if wanted == len(pool):
            whole = cls._ordered(pool, range(len(pool)))
            return cls(task=task, windows=whole, budget=budget, seed=seed)
        taken = strata.draw(pool, wanted, seed)
        return cls(task=task, windows=cls._ordered(pool, taken), budget=budget, seed=seed)

    @property
    def unit_count(self) -> int:
        """How many units the windows came from — fewer than the windows, which overlap."""
        return len({labelled.window.unit for labelled in self.windows})

    @property
    def mean_target(self) -> float:
        """The mean label of the sample: what a candidate that has learnt nothing yet answers.

        Summed in the sample's fixed order in double precision, so the value is the same on
        every machine and does not depend on where a run's arithmetic happens.

        Raises:
            EmptyLabelSampleError: If the sample holds no window.
        """
        if not self.windows:
            raise EmptyLabelSampleError("a sample without a window has no mean label")
        return sum(labelled.target for labelled in self.windows) / len(self.windows)

    @staticmethod
    def _ordered(
        pool: Sequence[LabelledWindow], taken: Iterable[int]
    ) -> tuple[LabelledWindow, ...]:
        """The drawn windows by unit and place, so a sample reads the same however it was drawn."""
        ordered = sorted(
            taken, key=lambda index: (str(pool[index].window.unit), pool[index].window.position)
        )
        return tuple(pool[index] for index in ordered)
