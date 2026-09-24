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
from emblema.evaluation.domain.exceptions import InvalidInnerHoldoutError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.run_labels import RunLabels
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
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
        holdout: How the tuning side is divided, for a selection run; ``None`` for any other.
    """

    task: TaskId
    budget: LabelBudget
    seed: int
    purpose: RunPurpose = RunPurpose.TUNING
    holdout: InnerHoldout | None = None


class DrawRunLabels:
    """Draws one budget of labels and labels the side the run will be scored on.

    The labels always come from the tuning side; which side the answers are scored on is what
    the run's purpose decides, and the decision is made here rather than by whoever calls. A
    tuning run is scored on the validation units and never asks for the frozen ones. A
    selection run never leaves the tuning side at all: the seed divides it, the budget is drawn
    from one part and the answers are scored on the other, so choosing a variant reads nothing
    the comparison it is chosen for is later read from. The final
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
            InvalidInnerHoldoutError: If a selection run was given no division of the tuning
                side, or another run was given one.
        """
        task = self._tasks.get(command.task)
        fitted, scored_units = self._sides(task, command)
        sample = self._draw(
            DrawLabelBudgetCommand(
                task=command.task, budget=command.budget, seed=command.seed, within=fitted
            )
        )
        windows = self._corpus.windows_of(task.manifest, scored_units)
        scored = task.labelled(windows, self._truth.truths_of(task.corpus, windows))
        return RunLabels(task=task, sample=sample, scored=scored)

    def _sides(
        self, task: DownstreamTask, command: DrawRunLabelsCommand
    ) -> tuple[frozenset[UnitKey] | None, frozenset[UnitKey]]:
        """The units the budget is drawn within, and the units the run answers.

        Raises:
            InvalidInnerHoldoutError: If the division and the purpose do not go together.
        """
        match command.purpose, command.holdout:
            case RunPurpose.SELECTION, InnerHoldout() as holdout:
                return holdout.divided(task.tuning_units, command.seed)
            case RunPurpose.SELECTION, None:
                raise InvalidInnerHoldoutError(
                    "a selection run needs a division of the tuning side"
                )
            case _, InnerHoldout():
                raise InvalidInnerHoldoutError(
                    f"a {command.purpose.value} run is scored outside the tuning side"
                )
            case RunPurpose.TUNING, None:
                return None, task.validation_units
            case _:
                opened = self._open(
                    OpenTestSplitCommand(task=task.task_id, purpose=command.purpose)
                )
                return None, opened.units
