from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction


@dataclass(frozen=True, kw_only=True)
class Adaptation:
    """One call the runtime answered, kept so a test can read what it was asked."""

    plan: AdaptationPlan
    task: DownstreamTask
    sample: LabelSample
    validation: tuple[LabelledWindow, ...]


class InMemoryAdaptationRuntime:
    """A runtime that learns nothing but the mean of the labels it was given, instantly.

    The candidate it makes answers every validation window with the sample's mean target — the
    trivial predictor a regression is measured against — whatever the plan asks for, so a use
    case built on it exercises the port's contract without a tensor: an answer per validation
    window in the order given, the identities that place the outcome on the curve, and a refusal
    of a sample from another task. The training loss it reports is the sample's variance, the
    loss the mean leaves, in every epoch; its one trainable parameter is the mean.
    """

    def __init__(self) -> None:
        self.adaptations: list[Adaptation] = []

    def adapt(
        self,
        plan: AdaptationPlan,
        task: DownstreamTask,
        sample: LabelSample,
        validation: Sequence[LabelledWindow],
    ) -> AdaptationOutcome:
        task.accept_sample(sample)
        self.adaptations.append(
            Adaptation(plan=plan, task=task, sample=sample, validation=tuple(validation))
        )
        targets = [labelled.target for labelled in sample.windows]
        mean = sum(targets) / len(targets)
        variance = sum((target - mean) ** 2 for target in targets) / len(targets)
        return AdaptationOutcome(
            plan=plan,
            task=task.task_id,
            budget=sample.budget,
            sample_seed=sample.seed,
            trainable_parameters=1,
            training_losses=(variance,) * plan.schedule.epochs,
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=mean)
                for labelled in validation
            ),
            seconds=0.0,
        )
