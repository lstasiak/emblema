from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.unit_lifetimes import UnitLifetimes


@dataclass(frozen=True, kw_only=True)
class DrawLabelBudgetCommand:
    """Request for the labels one point of a label-efficiency curve is allowed to learn from.

    Attributes:
        task: Task to draw from; it carries the strata the draw is spread over.
        budget: How many labelled windows.
        seed: Seed the draw is made under; repeats give the same windows.
    """

    task: TaskId
    budget: LabelBudget
    seed: int


class DrawLabelBudget:
    """Draws the labelled windows of one budget, from the tuning side alone.

    The validation and test sides are never read here: a budget is what a model learns from, and
    a draw that could reach the side a model is scored on would make every number after it
    unfalsifiable.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        lifetimes: UnitLifetimes,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._lifetimes = lifetimes

    def __call__(self, command: DrawLabelBudgetCommand) -> LabelSample:
        """Label every window of the tuning side, then draw the budget across the strata.

        Raises:
            TaskNotFoundError: If the task is unknown.
            UnknownUnitLifetimeError: If a tuning unit has no known failure time.
            UnlabelledWindowError: If a window reaches past the failure of its unit.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
            InvalidTargetBinsError: If it holds fewer windows than there are strata.
        """
        task = self._tasks.get(command.task)
        windows = self._corpus.windows_of(task.manifest, task.tuning_units)
        failed_at = self._lifetimes.failure_times(task.tuning_units)
        pool = [
            LabelledWindow(
                window=window,
                target=task.labels.target(failed_at=failed_at[window.unit], ends_at=window.ends_at),
            )
            for window in windows
        ]
        return LabelSample.drawn(task.task_id, pool, command.budget, task.strata, command.seed)
