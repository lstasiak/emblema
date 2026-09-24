from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidAdaptationOutcomeError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan


@dataclass(frozen=True, kw_only=True)
class AdaptationOutcome(ScoredOutcome):
    """What one cell of the curve produced: the candidate's answers on the validation side.

    A point on the curve is only a point if it can be placed: the plan, the task, the budget of
    labels and the seed they were drawn under travel with the predictions, so two outcomes are
    compared knowing what differs between them. The predictions are kept per window because the
    summaries a comparison needs are several and every one is arithmetic over them.

    Everything a scored outcome holds, and what only a network has to say about its run.

    Invariants: those of a scored outcome; one training loss per planned epoch, each finite and
    not negative; at least one trainable parameter; the labels are at least one window from at
    least one unit, no more units than windows, and as many windows as a counted budget asked
    for.

    Attributes:
        plan: How the candidate was made out of the backbone.
        task: Task the labels were drawn from and the windows are scored on.
        budget: How many labelled windows the candidate learnt from.
        sample_seed: Seed the labels were drawn under.
        labelled_windows: How many labelled windows the candidate learnt from — the budget
            resolved, since the largest budget is stated as everything rather than as a count.
        labelled_units: How many units the labelled windows came from. Windows of one unit
            overlap, so a budget of windows carries less than its count suggests, and the curve
            says beside each budget how many units stood behind it.
        trainable_parameters: How many weights the run could change.
        training_losses: Mean training loss per epoch, in the order trained.
    """

    plan: AdaptationPlan
    task: TaskId
    budget: LabelBudget
    sample_seed: int
    labelled_windows: int
    labelled_units: int
    trainable_parameters: int
    training_losses: tuple[float, ...]

    @property
    def optimiser_steps(self) -> int:
        """Optimiser steps the run took: its epochs over the batches its labelled windows make.

        The schedule's floor of steps lengthens a run in whole epochs, so what a cell cost in
        steps is read off the outcome rather than off the schedule.
        """
        return len(self.training_losses) * self.plan.schedule.steps_per_epoch(self.labelled_windows)

    def __post_init__(self) -> None:
        super().__post_init__()
        epochs = self.plan.schedule.epochs_over(self.labelled_windows)
        if len(self.training_losses) != epochs:
            raise InvalidAdaptationOutcomeError(
                f"{len(self.training_losses)} training losses reported for "
                f"{epochs} planned epochs over {self.labelled_windows} windows"
            )
        for loss in self.training_losses:
            if not isfinite(loss) or loss < 0.0:
                raise InvalidAdaptationOutcomeError(
                    f"a training loss must be finite and not negative, got {loss}"
                )
        if not 1 <= self.labelled_units <= self.labelled_windows:
            raise InvalidAdaptationOutcomeError(
                f"the labels must be at least one window from at least one unit and no more "
                f"units than windows, got {self.labelled_windows} windows from "
                f"{self.labelled_units} units"
            )
        if self.budget.windows is not None and self.labelled_windows != self.budget.windows:
            raise InvalidAdaptationOutcomeError(
                f"a budget of {self.budget.windows} windows resolved to {self.labelled_windows}"
            )
        if self.trainable_parameters < 1:
            raise InvalidAdaptationOutcomeError(
                f"a run must have trained a parameter, got {self.trainable_parameters}"
            )
