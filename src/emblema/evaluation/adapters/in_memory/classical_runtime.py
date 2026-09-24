from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


@dataclass(frozen=True, kw_only=True)
class Fitting:
    """One call the runtime answered, kept so a test can read what it was asked."""

    recipe: ClassicalRecipe
    task: DownstreamTask
    sample: LabelSample
    sources: tuple[FittingSource, ...]
    scored: tuple[LabelledWindow, ...]


class InMemoryClassicalRuntime:
    """A runtime that fits nothing but the mean of every label it was handed, instantly.

    The counterpart of the adaptation runtime that learns the mean, and the same argument for
    it: a campaign made of classical candidates then runs end to end against known numbers in
    the time it takes to call a function, and no test of the grid has to grow a tree to find out
    whether the grid works.

    Labels of the source tasks count towards the mean exactly as the target's do, so a test can
    tell a fit that ignored its sources from one that took them in.
    """

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self._store = store
        self.fittings: list[Fitting] = []

    def fit(
        self,
        recipe: ClassicalRecipe,
        task: DownstreamTask,
        sample: LabelSample,
        sources: Sequence[FittingSource],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        task.accept_sample(sample)
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        self.fittings.append(
            Fitting(
                recipe=recipe,
                task=task,
                sample=sample,
                sources=tuple(sources),
                scored=tuple(scored),
            )
        )
        mean = self._mean(task, sample, sources)
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(window=labelled.window, target=labelled.target, predicted=mean)
                for labelled in scored
            ),
            seconds=0.0,
            artifact=self._kept(mean) if retain else None,
        )

    @staticmethod
    def _mean(task: DownstreamTask, sample: LabelSample, sources: Sequence[FittingSource]) -> float:
        """The mean label over every draw the fit saw, read back in the target's own unit.

        Every task is divided by its own scheme's scale before the labels are pooled and the
        mean is multiplied back by the target's, exactly as a runtime that really fits does:
        a fake that pooled cycles with hours would answer a number no task asks for.
        """
        scale = task.label_scheme().scale
        drawn = [
            (sample, scale),
            *((source.sample, source.task.label_scheme().scale) for source in sources),
        ]
        targets = [
            labelled.target / of_task for labels, of_task in drawn for labelled in labels.windows
        ]
        return sum(targets) / len(targets) * scale

    def _kept(self, mean: float) -> ArtifactRef:
        """The mean this runtime fitted, stored, so a retained cell names real bytes.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        return self._store.put(repr(mean).encode())
