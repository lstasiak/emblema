from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.application.use_cases.open_test_split import (
    OpenTestSplit,
    OpenTestSplitCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.run_labels import RunLabels
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.ground_truth import GroundTruth


@dataclass(frozen=True, kw_only=True)
class DrawRunLabelsCommand:
    """Which task a run is over, how many labels it may learn from, and what it is for.

    Attributes:
        task: Task to learn and to be scored on.
        budget: How many labelled windows the run learns from.
        seed: Seed the draw is made under; repeats give the same windows.
        purpose: What the run is for, which decides which side it is scored on.
    """

    task: TaskId
    budget: LabelBudget
    seed: int
    purpose: RunPurpose = RunPurpose.TUNING


class DrawRunLabels:
    """Draws one budget of labels and labels the side the run will be scored on.

    The labels always come from the tuning side; which side the answers are scored on is what
    the run's purpose decides, and the decision is made here rather than by whoever calls. A
    tuning run is scored on the validation units and never asks for the frozen ones. The final
    run asks for them through the operation that records the asking, so that the claim of a
    single reading of the test set rests on a record rather than on a convention.

    The scoring windows are labelled by the same scheme the sample was, so a candidate is scored
    against the targets it was taught to predict and not against a scheme chosen at scoring time.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        truth: GroundTruth,
        draw_label_budget: DrawLabelBudget,
        open_test_split: OpenTestSplit,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._truth = truth
        self._draw = draw_label_budget
        self._open = open_test_split

    def __call__(self, command: DrawRunLabelsCommand) -> RunLabels:
        """The task, the budget drawn from its tuning side, and the labelled side it answers.

        Raises:
            TaskNotFoundError: If the task is unknown.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a window of either
                side.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
        """
        task = self._tasks.get(command.task)
        sample = self._draw(
            DrawLabelBudgetCommand(task=command.task, budget=command.budget, seed=command.seed)
        )
        windows = self._corpus.windows_of(task.manifest, self._scored_units(task, command.purpose))
        scored = task.labelled(windows, self._truth.truths_of(task.corpus, windows))
        return RunLabels(task=task, sample=sample, scored=scored)

    def _scored_units(self, task: DownstreamTask, purpose: RunPurpose) -> frozenset[UnitKey]:
        """Which units the run answers: the validation side, or the frozen one for the last run."""
        if purpose is RunPurpose.TUNING:
            return task.validation_units
        return self._open(OpenTestSplitCommand(task=task.task_id, purpose=purpose)).units
