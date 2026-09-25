from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.patching.patch_plan import PatchPlan
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


@dataclass(frozen=True, kw_only=True)
class PatchTraining:
    """One call the runtime answered, kept so a test can read what it was asked."""

    plan: PatchPlan
    task: DownstreamTask
    sample: LabelSample
    scored: tuple[LabelledWindow, ...]


class InMemoryPatchRuntime:
    """A runtime that learns nothing but the mean of the labels it was given, instantly.

    Every scored window is answered with the sample's mean target, whatever the plan asks for,
    so a use case built on it exercises the port's contract without a tensor. Given a store it
    keeps the mean it learnt, so a campaign assembled over it names artifacts that really exist.
    """

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._store = store
        self.trainings: list[PatchTraining] = []

    def train(
        self,
        plan: PatchPlan,
        task: DownstreamTask,
        sample: LabelSample,
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        task.accept_sample(sample)
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        self.trainings.append(
            PatchTraining(plan=plan, task=task, sample=sample, scored=tuple(scored))
        )
        mean = sample.mean_target
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=mean)
                for labelled in scored
            ),
            seconds=0.0,
            artifact=self._kept(mean) if retain else None,
        )

    def _kept(self, mean: float) -> ArtifactRef:
        """The mean this runtime learnt, stored, so a retained cell names real bytes.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it trained and was given no store"
            )
        return self._store.put(repr(mean).encode())
