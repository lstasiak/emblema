from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_run_labels import (
    DrawRunLabels,
    DrawRunLabelsCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.ports.adaptation_runtime import AdaptationRuntime


@dataclass(frozen=True, kw_only=True)
class RunAdaptationCommand:
    """One cell of the label-efficiency curve: a plan, a budget of labels, and the draw's seed.

    Attributes:
        task: Task to learn and to be scored on.
        plan: How the candidate is made out of the backbone.
        budget: How many labelled windows it learns from.
        sample_seed: Seed the labels are drawn under; the plan carries the seed of the learning.
        purpose: What the run is for, which decides which side it is scored on.
        holdout: How the tuning side is divided, for a selection run; ``None`` for any other.
        retain: Whether the candidate this run fits is kept as an artifact.
    """

    task: TaskId
    plan: AdaptationPlan
    budget: LabelBudget
    sample_seed: int
    purpose: RunPurpose = RunPurpose.TUNING
    retain: bool = False
    holdout: InnerHoldout | None = None


class RunAdaptation:
    """Teaches a candidate made out of a backbone from one budget of labels, and scores it.

    Which labels those are is not this use case's business: drawing them is the same for every
    candidate a campaign compares, so it is done in one place and this one is left with the
    single step that is its own.
    """

    def __init__(self, draw_run_labels: DrawRunLabels, runtime: AdaptationRuntime) -> None:
        self._labels = draw_run_labels
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
        labels = self._labels(
            DrawRunLabelsCommand(
                task=command.task,
                budget=command.budget,
                seed=command.sample_seed,
                purpose=command.purpose,
                holdout=command.holdout,
            )
        )
        return self._runtime.adapt(
            command.plan, labels.task, labels.sample, labels.scored, retain=command.retain
        )
