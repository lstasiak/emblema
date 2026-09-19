from dataclasses import dataclass
from math import isfinite, sqrt

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidAdaptationOutcomeError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.unit_error import UnitError
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class AdaptationOutcome:
    """What one cell of the curve produced: the candidate's answers on the validation side.

    A point on the curve is only a point if it can be placed: the plan, the task, the budget of
    labels and the seed they were drawn under travel with the predictions, so two outcomes are
    compared knowing what differs between them. The predictions are kept per window because the
    summaries a comparison needs are several and every one is arithmetic over them.

    Invariants: at least one prediction, no window predicted twice; one training loss per
    planned epoch, each finite and not negative; at least one trainable parameter; the labels
    are at least one window from at least one unit, no more units than windows, and as many
    windows as a counted budget asked for; the time taken is not negative.

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
        predictions: The candidate's answer for every validation window, in block order.
        seconds: What the run took, learning and scoring together.
    """

    plan: AdaptationPlan
    task: TaskId
    budget: LabelBudget
    sample_seed: int
    labelled_windows: int
    labelled_units: int
    trainable_parameters: int
    training_losses: tuple[float, ...]
    predictions: tuple[WindowPrediction, ...]
    seconds: float

    def __post_init__(self) -> None:
        if not self.predictions:
            raise InvalidAdaptationOutcomeError("an outcome must predict at least one window")
        places = [(str(p.window.unit), p.window.position) for p in self.predictions]
        if len(set(places)) != len(places):
            raise InvalidAdaptationOutcomeError("a window is predicted twice")
        if len(self.training_losses) != self.plan.schedule.epochs:
            raise InvalidAdaptationOutcomeError(
                f"{len(self.training_losses)} training losses reported for "
                f"{self.plan.schedule.epochs} planned epochs"
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
        if not isfinite(self.seconds) or self.seconds < 0.0:
            raise InvalidAdaptationOutcomeError(
                f"seconds must be finite and not negative, got {self.seconds}"
            )

    @property
    def rmse(self) -> float:
        """The root mean squared error over every validation window."""
        return sqrt(sum(p.squared_error for p in self.predictions) / len(self.predictions))

    def by_unit(self) -> tuple[UnitError, ...]:
        """The error per validation unit, in unit order — what a paired comparison resamples."""
        return UnitError.per_unit(self.predictions)
