from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_run_labels import (
    DrawRunLabels,
    DrawRunLabelsCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.patch_runtime import PatchRuntime


@dataclass(frozen=True, kw_only=True)
class RunPatchTrainingCommand:
    """One cell of the curve for a patch model: a plan, a budget of labels, and the draw's seed.

    Attributes:
        task: Task to learn and to be scored on.
        plan: How the model is shaped and how long it learns.
        budget: How many labelled windows it learns from.
        sample_seed: Seed the labels are drawn under; the plan carries the seed of the learning.
        purpose: What the run is for, which decides which side it is scored on.
        retain: Whether the model this run trains is kept as an artifact.
        holdout: How the tuning side is divided, for a selection run; ``None`` for any other.
    """

    task: TaskId
    plan: PatchPlan
    budget: LabelBudget
    sample_seed: int
    purpose: RunPurpose = RunPurpose.TUNING
    retain: bool = False
    holdout: InnerHoldout | None = None


class RunPatchTraining:
    """Trains a patch model from nothing on one budget of labels, and scores it.

    The labels are drawn by the step every candidate shares, from the same side under the same
    seed, so a cell of the grid means the same thing whichever kind of candidate stood in it.
    """

    def __init__(self, draw_run_labels: DrawRunLabels, runtime: PatchRuntime) -> None:
        self._labels = draw_run_labels
        self._runtime = runtime

    def __call__(self, command: RunPatchTrainingCommand) -> ScoredOutcome:
        """Draw, train, score; return the model's answer for every scored window.

        Raises:
            TaskNotFoundError: If the task is unknown.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a window of either
                side.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
            InvalidPatchModelSpecError: If a window of the task is shorter than one patch.
            CandidateNotRetainableError: If the run was to keep what it trained and the runtime
                has nowhere to keep it.
        """
        labels = self._labels(
            DrawRunLabelsCommand(
                task=command.task,
                budget=command.budget,
                seed=command.sample_seed,
                purpose=command.purpose,
                holdout=command.holdout,
            )
        )
        return self._runtime.train(
            command.plan, labels.task, labels.sample, labels.scored, retain=command.retain
        )
