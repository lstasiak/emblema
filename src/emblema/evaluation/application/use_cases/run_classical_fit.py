from dataclasses import dataclass

from emblema.evaluation.application.use_cases.draw_label_budget import (
    DrawLabelBudget,
    DrawLabelBudgetCommand,
)
from emblema.evaluation.application.use_cases.draw_run_labels import (
    DrawRunLabels,
    DrawRunLabelsCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.classical_outcome import ClassicalOutcome
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.classical_runtime import ClassicalRuntime
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository


@dataclass(frozen=True, kw_only=True)
class RunClassicalFitCommand:
    """One cell of the curve for a candidate with no weights: a recipe, a budget, and a seed.

    Attributes:
        task: Task to learn and to be scored on.
        recipe: What the candidate reads, how it fits, and which other tasks it draws on.
        budget: How many labelled windows it learns from, per task it learns from.
        sample_seed: Seed the labels are drawn under; the recipe carries the seed of the fit.
        purpose: What the run is for, which decides which side it is scored on.
        retain: Whether the candidate this run fits is kept as an artifact.
    """

    task: TaskId
    recipe: ClassicalRecipe
    budget: LabelBudget
    sample_seed: int
    purpose: RunPurpose = RunPurpose.TUNING
    retain: bool = False


class RunClassicalFit:
    """Fits a candidate that starts from no weights on one budget of labels, and scores it.

    The counterpart of adapting a backbone, drawing the same labels from the same side under the
    same seed, so that a cell of the grid means the same thing whichever kind of candidate stood
    in it. What differs is the source tasks: a recipe that names them is fitted on their labels
    as well, which is how a classical method answers the question about transfer.

    Every source contributes a budget of the same size, drawn under its own task's strata and
    under the run's seed. That it is the same size is a decision, not an omission: a second dial
    for how much a source contributes would be a dial nobody recorded and everybody could turn
    after seeing the numbers, so what a transfer candidate saw follows from the cell's
    coordinates and from the tasks the recipe names.
    """

    def __init__(
        self,
        tasks: DownstreamTaskRepository,
        draw_run_labels: DrawRunLabels,
        draw_label_budget: DrawLabelBudget,
        runtime: ClassicalRuntime,
    ) -> None:
        self._tasks = tasks
        self._labels = draw_run_labels
        self._draw = draw_label_budget
        self._runtime = runtime

    def __call__(self, command: RunClassicalFitCommand) -> ClassicalOutcome:
        """Draw, fit, score; return the candidate's answer for every scored window.

        Raises:
            TaskNotFoundError: If the task, or one the recipe draws on, is unknown.
            FrozenTestSplitClosedError: If a run that is not the final one was to be scored on
                the frozen side.
            UnknownGroundTruthError: If the ground truth says nothing about a window of either
                side, or about the corpus of a source task.
            InvalidLabelBudgetError: If a tuning side holds fewer windows than asked for.
            CandidateNotRetainableError: If the run was to keep what it fitted and the runtime
                has nowhere to keep it.
        """
        labels = self._labels(
            DrawRunLabelsCommand(
                task=command.task,
                budget=command.budget,
                seed=command.sample_seed,
                purpose=command.purpose,
            )
        )
        sources = tuple(
            self._contribution(source, command.budget, command.sample_seed)
            for source in command.recipe.sources
        )
        return self._runtime.fit(
            command.recipe,
            labels.task,
            labels.sample,
            sources,
            labels.scored,
            retain=command.retain,
        )

    def _contribution(self, task: TaskId, budget: LabelBudget, seed: int) -> FittingSource:
        """One source task and the labels it lends, drawn from its own tuning side."""
        return FittingSource(
            task=self._tasks.get(task),
            sample=self._draw(DrawLabelBudgetCommand(task=task, budget=budget, seed=seed)),
        )
