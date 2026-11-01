import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.detection.feature_detector import FeatureDetector
from emblema.evaluation.adapters.detection.feature_detectors import FeatureDetectors
from emblema.evaluation.adapters.torch.backbone_factory import BackboneFactory
from emblema.evaluation.adapters.torch.frozen_embedding import FrozenEmbedding
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
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
from emblema.evaluation.domain.transfer.encoder_setting import CellSet
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.arrays.laid_window import LaidWindow
from emblema.shared.adapters.arrays.raw_tokens import RawTokens
from emblema.shared.adapters.arrays.window_layout import WindowLayout
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore

# Windows are embedded this many at a time; a detector takes no step, so the batch is a matter
# of memory alone.
_BATCH = 64


class TorchDetectionRuntime:
    """Fits a detector over the states of a frozen encoder, or over its cells, in this process.

    Every window of the fitting side and of the scored side is laid out as the encoder was
    trained to read it, embedded once under no gradient, and the detector is fitted and scored
    over the states on the host in double precision; asked for the cells themselves, the runtime
    lays every window out over the corpus's channels and hands the cells to the detector with no
    encoder between, the bar of the input itself. The distance's arithmetic is numpy's, as it is
    for the summaries, so the detectors over states, cells and summaries differ in what they read
    and in nothing else. The seconds an outcome reports start once the block is at hand.
    """

    def __init__(
        self,
        backbones: BackboneFactory,
        blocks: PublishedCorpusBlocks,
        *,
        device: str,
        store: ArtifactStore | None = None,
    ) -> None:
        self._backbones = backbones
        self._blocks = blocks
        self._device = device
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
        if not recipe.features.lays_out_cells:
            raise UnsupportedDetectorFeaturesError(
                f"this runtime reads an encoder's states or its cells, not {recipe.features}"
            )
        if not fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        manifest = self._blocks.manifest_of(task.manifest)
        block = self._blocks.block_of(manifest)
        fitted_windows = block.at([window.position for window in fitting])
        scored_windows = block.at([labelled.window.position for labelled in scored])
        started = time.perf_counter()
        rows = self._rows_of(recipe, manifest)
        per_unit = recipe.reference is ScoreReference.OWN_UNIT
        detector = FeatureDetectors.fitted(
            recipe,
            rows(fitted_windows),
            units=[str(window.unit) for window in fitting] if per_unit else None,
        )
        answers = detector.scores(
            rows(scored_windows),
            units=[str(labelled.window.unit) for labelled in scored] if per_unit else None,
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

    def _rows_of(
        self, recipe: DetectorRecipe, manifest: PublishedCorpusManifest
    ) -> Callable[[Sequence[TokenWindow]], NDArray[np.float64]]:
        """What the detector reads a window as: the encoder's state, or the cells themselves.

        Raises:
            UnsupportedDetectorFeaturesError: If the cells are asked of a backbone that reads
                one token per reading, which lays no cells out.
        """
        if recipe.reading is None:
            if not self._backbones.bins:
                raise UnsupportedDetectorFeaturesError(
                    "the backbone reads one token per reading, so there are no cells to read"
                )
            cells = self._corpus_layout(self._backbones.bins, manifest)
            return lambda windows: np.stack([laid.features.ravel() for laid in cells.lay(windows)])
        reading = recipe.reading
        layout = self._layout(reading, manifest)
        torch.manual_seed(recipe.seed)
        embedding = FrozenEmbedding.under(
            reading, self._backbones, vocabulary_size=len(manifest.channels), layout=layout
        ).to(self._device)
        embedding.eval()
        return lambda windows: self._states(embedding, layout, windows)

    def _layout(self, reading: EncoderReading, manifest: PublishedCorpusManifest) -> WindowLayout:
        """How the windows are laid out: as the backbone read them, unless the setting says.

        The encoder at its initialisation reads the backbone's layout too, cells where the
        backbone read cells, so the two arms differ in their weights and in nothing else.
        """
        bins = reading.encoder.bins
        if bins is None:
            return BinnedTokens(self._backbones.bins) if self._backbones.bins else RawTokens()
        if reading.encoder.cells is CellSet.WINDOW:
            return BinnedTokens(bins)
        return self._corpus_layout(bins, manifest)

    @staticmethod
    def _corpus_layout(bins: int, manifest: PublishedCorpusManifest) -> BinnedTokens:
        """Every window laid out over the corpus's channels, so every window has the same cells."""
        observed = [channel for channel in manifest.channels if channel.corpus == manifest.corpus]
        return BinnedTokens(
            bins,
            channels=[channel.channel_id for channel in observed if not channel.timeless],
            statics=[channel.channel_id for channel in observed if channel.timeless],
        )

    def _states(
        self, embedding: FrozenEmbedding, layout: WindowLayout, windows: Sequence[TokenWindow]
    ) -> NDArray[np.float64]:
        """One state per window, on the host in double precision, in the order given."""
        laid: list[LaidWindow] = layout.lay(windows)
        states = []
        for start in range(0, len(laid), _BATCH):
            batch = TokenTensors.of(layout.batch(laid[start : start + _BATCH])).to(self._device)
            # Moved before it is widened: an accelerator without double precision hands back
            # zeros when asked to do both at once.
            states.append(embedding.embed(batch).to("cpu").to(torch.float64).numpy())
        return np.concatenate(states)

    def _kept(self, detector: FeatureDetector, task: DownstreamTask) -> ArtifactRef:
        """The detector this fit produced, stored under a manifest, for the campaign.

        The states it was fitted over are the measured form; the encoder they came from is named
        by the campaign's candidate, not stored again here.

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
