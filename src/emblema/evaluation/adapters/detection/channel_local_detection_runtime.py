import time
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.detection.channel_departures import ChannelDepartures
from emblema.evaluation.adapters.detection.channel_local_detector import ChannelLocalDetector
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    InvalidTaughtEventsError,
    UnsupportedDetectorFeaturesError,
)
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class ChannelLocalDetectionRuntime:
    """Fits a detector over each channel's departure from its own recent windows, in this process.

    The bar of a change local to a few channels, with no model: every window of either side is
    laid out on six spans per channel, as the encoder candidates read it, and each channel is
    read against the same channel over the windows handed in that end before it, the fitting
    side's and the scored side's alike, so the history never reaches a window the caller did not
    read. Nothing is learnt from a label: the fitting side sets the floor of the spreads and the
    alarm. The seconds an outcome reports start once the block is at hand.
    """

    FEATURES: ClassVar[frozenset[DetectorFeatures]] = frozenset(
        {DetectorFeatures.CHANNEL_DEPARTURES}
    )
    # Six spans, as the published cells the encoder candidates and the channel-local report
    # read, so a span of the telemetry's hour is ten minutes.
    SPANS: ClassVar[int] = 6
    CHUNK: ClassVar[int] = 512

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
        taught: Sequence[LabelledWindow] = (),
        retain: bool,
    ) -> ScoredOutcome:
        if recipe.features not in self.FEATURES:
            raise UnsupportedDetectorFeaturesError(
                "this runtime reads each channel's departure from its history, "
                f"not {recipe.features}"
            )
        if taught:
            raise InvalidTaughtEventsError(
                f"a detector over {recipe.features} departs from its history; it learns from no "
                "event"
            )
        if not fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        history = recipe.read_history()
        read = ReadCorpus.every(self._blocks, (task,))[task.manifest]
        windows = (*fitting, *(labelled.window for labelled in scored))
        started = time.perf_counter()
        departures = ChannelDepartures.laid(
            history,
            self._cells(read, windows),
            ends_at=[window.ends_at for window in windows],
            units=[str(window.unit) for window in windows],
            window_length=read.manifest.window_length,
        )
        on_fit = np.arange(len(fitting), dtype=np.int64)
        on_scored = np.arange(len(fitting), len(windows), dtype=np.int64)
        floor = departures.floor_over(on_fit)
        per_unit = recipe.reference is ScoreReference.OWN_UNIT
        detector = ChannelLocalDetector.fitted_on(
            departures.of(on_fit, floor),
            departures.present(on_fit),
            history=history,
            floor=floor,
            contamination=recipe.contamination,
            units=[str(window.unit) for window in fitting] if per_unit else None,
        )
        answers = detector.scores(
            departures.of(on_scored, floor),
            departures.present(on_scored),
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

    def _cells(
        self, read: ReadCorpus, windows: Sequence[TaskWindow]
    ) -> Iterator[NDArray[np.float64]]:
        """``[windows, channels, spans, features]`` chunk by chunk over the corpus's channels.

        The corpus's static features are laid as the layout requires and left out, having no
        history to depart from.
        """
        manifest = read.manifest
        own = [channel for channel in manifest.channels if channel.corpus == manifest.corpus]
        statics = [channel.channel_id for channel in own if channel.timeless]
        layout = BinnedTokens(
            self.SPANS,
            channels=[channel.channel_id for channel in own if not channel.timeless],
            statics=statics,
        )
        rows = len(layout.channels)
        if not rows:
            raise UnsupportedDetectorFeaturesError(
                f"{manifest.corpus} measures no channel over time, so no channel has a history"
            )
        for start in range(0, len(windows), self.CHUNK):
            laid = layout.lay(read.windows(windows[start : start + self.CHUNK]))
            yield np.stack(
                [
                    each.features[len(statics) :].reshape(self.SPANS, rows, -1).transpose(1, 0, 2)
                    for each in laid
                ]
            )

    def _kept(self, detector: ChannelLocalDetector, task: DownstreamTask) -> ArtifactRef:
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
