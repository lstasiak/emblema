from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidInnerHoldoutError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.ground_truth import GroundTruth


@dataclass(frozen=True, kw_only=True)
class DrawLabelBudgetCommand:
    """Request for the labels one point of a label-efficiency curve is allowed to learn from.

    Attributes:
        task: Task to draw from; it carries the strata the draw is spread over.
        budget: How many labelled windows.
        seed: Seed the draw is made under; repeats give the same windows.
        within: The tuning units the draw is confined to, where part of the side is held out
            to score on; ``None`` for the whole tuning side.
    """

    task: TaskId
    budget: LabelBudget
    seed: int
    within: frozenset[UnitKey] | None = None


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
        truth: GroundTruth,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._truth = truth

    def __call__(self, command: DrawLabelBudgetCommand) -> LabelSample:
        """Label every window of the tuning side, then draw the budget across the strata.

        Raises:
            TaskNotFoundError: If the task is unknown.
            UnknownGroundTruthError: If nothing is known about the task's corpus, or about a
                tuning window of it.
            UnlabelledWindowError: If a window reaches past the failure of its unit.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
            InvalidTargetBinsError: If it holds fewer windows than there are strata.
            InvalidInnerHoldoutError: If the draw is confined to units outside the tuning side.
        """
        task = self._tasks.get(command.task)
        units = task.tuning_units if command.within is None else command.within
        if not units <= task.tuning_units:
            raise InvalidInnerHoldoutError(
                f"a draw from task {task.task_id} is confined to units outside its tuning side"
            )
        windows = self._corpus.windows_of(task.manifest, units)
        pool = task.labelled(windows, self._truth.truths_of(task.corpus, windows))
        return LabelSample.drawn(
            task.task_id, pool, command.budget, task.stratification(), command.seed
        )
