from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.ports.adaptation_runtime import AdaptationRuntime
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.ground_truth import GroundTruth


@dataclass(frozen=True, kw_only=True)
class RunAdaptationCommand:
    """One cell of the label-efficiency curve: a plan, a budget of labels, and the draw's seed.

    Attributes:
        task: Task to learn and to be scored on.
        plan: How the candidate is made out of the backbone.
        budget: How many labelled windows it learns from.
        sample_seed: Seed the labels are drawn under; the plan carries the seed of the learning.
    """

    task: TaskId
    plan: AdaptationPlan
    budget: LabelBudget
    sample_seed: int


class RunAdaptation:
    """Draws one budget of labels, teaches a candidate from them, scores it on the validation side.

    A tuning run: the labels come from the tuning side alone and the answers are scored on the
    validation side alone, and the frozen test side is never asked for. The validation windows
    are labelled here, by the same scheme the sample was, so a candidate is scored against the
    targets it was taught to predict and not against a scheme chosen at scoring time.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        truth: GroundTruth,
        draw_label_budget: DrawLabelBudget,
        runtime: AdaptationRuntime,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._truth = truth
        self._draw = draw_label_budget
        self._runtime = runtime

    def __call__(self, command: RunAdaptationCommand) -> AdaptationOutcome:
        """Draw, learn, score; return the candidate's answer for every validation window.

        Raises:
            TaskNotFoundError: If the task is unknown.
            UnknownGroundTruthError: If the ground truth says nothing about a window of either
                side.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
            UnknownBackboneError: If the plan names pretrained weights the runtime cannot supply.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
        """
        task = self._tasks.get(command.task)
        sample = self._draw(
            DrawLabelBudgetCommand(
                task=command.task, budget=command.budget, seed=command.sample_seed
            )
        )
        windows = self._corpus.windows_of(task.manifest, task.validation_units)
        validation = task.labelled(windows, self._truth.truths_of(windows))
        return self._runtime.adapt(command.plan, task, sample, validation)
