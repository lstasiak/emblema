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
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.run_purpose import RunPurpose
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
        purpose: What the run is for, which decides which side it is scored on.
        retain: Whether the candidate this run fits is kept as an artifact.
    """

    task: TaskId
    plan: AdaptationPlan
    budget: LabelBudget
    sample_seed: int
    purpose: RunPurpose = RunPurpose.TUNING
    retain: bool = False


class RunAdaptation:
    """Draws one budget of labels, teaches a candidate from them, scores it on the right side.

    The labels always come from the tuning side; which side the answers are scored on is what
    the run's purpose decides, and the decision is made here rather than by whoever calls. A
    tuning run is scored on the validation units and never asks for the frozen ones. The final
    run asks for them through the operation that records the asking, so that the claim of a
    single reading of the test set rests on a record rather than on a convention.

    The scoring windows are labelled here, by the same scheme the sample was, so a candidate is
    scored against the targets it was taught to predict and not against a scheme chosen at
    scoring time.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        corpus: CorpusWindows,
        truth: GroundTruth,
        draw_label_budget: DrawLabelBudget,
        open_test_split: OpenTestSplit,
        runtime: AdaptationRuntime,
    ) -> None:
        self._tasks = tasks
        self._corpus = corpus
        self._truth = truth
        self._draw = draw_label_budget
        self._open = open_test_split
        self._runtime = runtime

    def __call__(self, command: RunAdaptationCommand) -> AdaptationOutcome:
        """Draw, learn, score; return the candidate's answer for every scored window.

        Raises:
            TaskNotFoundError: If the task is unknown.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a window of either
                side.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
            UnknownBackboneError: If the plan names pretrained weights the runtime cannot supply.
            LoraTargetNotFoundError: If the plan's low-rank updates name a layer the backbone
                does not have.
            CandidateNotRetainableError: If the run was to keep what it fitted and the runtime
                has nowhere to keep it.
        """
        task = self._tasks.get(command.task)
        sample = self._draw(
            DrawLabelBudgetCommand(
                task=command.task, budget=command.budget, seed=command.sample_seed
            )
        )
        windows = self._corpus.windows_of(task.manifest, self._scored_units(task, command.purpose))
        scored = task.labelled(windows, self._truth.truths_of(task.corpus, windows))
        return self._runtime.adapt(command.plan, task, sample, scored, retain=command.retain)

    def _scored_units(self, task: DownstreamTask, purpose: RunPurpose) -> frozenset[UnitKey]:
        """Which units the run answers: the validation side, or the frozen one for the last run."""
        if purpose is RunPurpose.TUNING:
            return task.validation_units
        return self._open(OpenTestSplitCommand(task=task.task_id, purpose=purpose)).units
