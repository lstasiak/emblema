from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


@dataclass(frozen=True, kw_only=True)
class Detection:
    """One call the runtime answered, kept so a test can read what it was asked."""

    recipe: DetectorRecipe
    task: DownstreamTask
    fitting: tuple[TaskWindow, ...]
    scored: tuple[LabelledWindow, ...]


class InMemoryDetectionRuntime:
    """A runtime that scores a window by how far its end lies past the fitting windows', instantly.

    The one fact of the fitting side a detector could read without a token is where its windows
    end, so the alarm is set at the latest end the fit saw and a scored window's score is how
    far past it the window ends: a window inside the span the fit covered scores at or below
    zero and one beyond it raises an alarm. Enough for a campaign of detectors to run end to
    end against known numbers, and for a test to tell a runtime that read the fitting side from
    one that did not.
    """

    FORMAT: ClassVar[str] = "latest-end"

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._kept_candidates = None if store is None else KeptCandidates(store)
        self.detections: list[Detection] = []

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        if not fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        self.detections.append(
            Detection(recipe=recipe, task=task, fitting=tuple(fitting), scored=tuple(scored))
        )
        alarm = max(window.ends_at for window in fitting)
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window,
                    target=labelled.target,
                    predicted=labelled.window.ends_at - alarm,
                )
                for labelled in scored
            ),
            seconds=0.0,
            artifact=self._kept(alarm, task) if retain else None,
        )

    def _kept(self, alarm: float, task: DownstreamTask) -> ArtifactRef:
        """The alarm this runtime set, under a manifest, so a retained cell names real bytes.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        return self._kept_candidates.keep(
            CandidateKind.DETECTOR,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(self.FORMAT, repr(alarm).encode()),
        )
