from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.exceptions import CandidateNotRetainableError
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


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

    Given a store it can be asked to keep what it fitted, which for this runtime is the mean it
    learnt, under a manifest as every runtime keeps a candidate: a campaign assembled over it
    then names artifacts that really exist, really hash to what the message says and read as a
    kept candidate, rather than references to nothing.
    """

    FORMAT: ClassVar[str] = "learnt-mean"

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._kept_candidates = None if store is None else KeptCandidates(store)
        self.adaptations: list[Adaptation] = []

    def adapt(
        self,
        plan: AdaptationPlan,
        task: DownstreamTask,
        sample: LabelSample,
        validation: Sequence[LabelledWindow],
        *,
        retain: bool,
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
            labelled_windows=len(sample.windows),
            labelled_units=sample.unit_count,
            trainable_parameters=1,
            training_losses=(variance,) * plan.schedule.epochs,
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=mean)
                for labelled in validation
            ),
            seconds=0.0,
            artifact=self._kept(mean, task) if retain else None,
        )

    def _kept(self, mean: float, task: DownstreamTask) -> ArtifactRef:
        """The mean this runtime learnt, under a manifest, so a retained cell names real bytes.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        return self._kept_candidates.keep(
            CandidateKind.NEURAL,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(self.FORMAT, repr(mean).encode()),
        )
