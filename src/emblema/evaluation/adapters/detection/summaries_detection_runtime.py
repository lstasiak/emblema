import time
from collections.abc import Sequence

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.detection.feature_detector import FeatureDetector
from emblema.evaluation.adapters.detection.feature_detectors import FeatureDetectors
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    UnsupportedDetectorFeaturesError,
)
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class SummariesDetectionRuntime:
    """Fits a detector over the statistics of every channel of a window, in this process.

    The classical bar for a representation: the same table the trees read, standardised over
    the fitting side and scored by the distance the recipe names, to the nearest fitting windows
    or whitened; or that table's counts of readings alone, the bar of the sampling. Nothing here
    reaches an encoder, so a process without the machine-learning stack can run it. The seconds
    an outcome reports start once the block is at hand.
    """

    FEATURES = frozenset({DetectorFeatures.SUMMARIES, DetectorFeatures.READING_COUNTS})

    def __init__(
        self, blocks: PublishedCorpusBlocks, *, store: ArtifactStore | None = None
    ) -> None:
        self._blocks = blocks
        self._kept_candidates = None if store is None else KeptCandidates(store)

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        if recipe.features not in self.FEATURES:
            raise UnsupportedDetectorFeaturesError(
                "this runtime reads the summaries per channel or their counts, "
                f"not {recipe.features}"
            )
        if not fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        read = ReadCorpus.every(self._blocks, (task,))[task.manifest]
        features = PerChannelFeatures(read.channels)
        rows_of = (
            features.counts_of
            if recipe.features is DetectorFeatures.READING_COUNTS
            else features.of
        )
        per_unit = recipe.reference is ScoreReference.OWN_UNIT
        started = time.perf_counter()
        detector = FeatureDetectors.fitted(
            recipe,
            rows_of(read.windows(fitting)),
            units=[str(window.unit) for window in fitting] if per_unit else None,
        )
        answers = detector.scores(
            rows_of(read.windows([w.window for w in scored])),
            units=[str(w.window.unit) for w in scored] if per_unit else None,
        )
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window, target=labelled.target, predicted=float(answer)
                )
                for labelled, answer in zip(scored, answers.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(detector, task) if retain else None,
        )

    def _kept(self, detector: FeatureDetector, task: DownstreamTask) -> ArtifactRef:
        """The detector this fit produced, stored under a manifest, for the campaign.

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
            measured=detector.kept(),
        )
