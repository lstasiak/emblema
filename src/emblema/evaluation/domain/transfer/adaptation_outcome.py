from dataclasses import dataclass
from math import isfinite, sqrt

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidAdaptationOutcomeError
from emblema.evaluation.domain.identifiers import UnitKey
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
    planned epoch, each finite and not negative; at least one trainable parameter; the time
    taken is not negative.

    Attributes:
        plan: How the candidate was made out of the backbone.
        task: Task the labels were drawn from and the windows are scored on.
        budget: How many labelled windows the candidate learnt from.
        sample_seed: Seed the labels were drawn under.
        trainable_parameters: How many weights the run could change.
        training_losses: Mean training loss per epoch, in the order trained.
        predictions: The candidate's answer for every validation window, in block order.
        seconds: What the run took, learning and scoring together.
    """

    plan: AdaptationPlan
    task: TaskId
    budget: LabelBudget
    sample_seed: int
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
        units: dict[UnitKey, list[WindowPrediction]] = {}
        for prediction in self.predictions:
            units.setdefault(prediction.window.unit, []).append(prediction)
        return tuple(
            UnitError.of(unit, units[unit]) for unit in sorted(units, key=lambda unit: str(unit))
        )
