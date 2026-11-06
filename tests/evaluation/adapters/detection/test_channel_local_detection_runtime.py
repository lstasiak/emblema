from dataclasses import replace
from pathlib import Path

import pytest

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.detection.channel_local_detection_runtime import (
    ChannelLocalDetectionRuntime,
)
from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.exceptions import (
    OverlappingWindowsError,
    UnsupportedDetectorFeaturesError,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.evaluation.support import labelled, task, window
from tests.support.published import publish


def test_a_corpus_measuring_no_channel_over_time_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    blocks = PublishedCorpusBlocks(store, tmp_path / "workspace")
    manifest = publish(store, tmp_path / "scratch").manifest
    (corpus,) = ReadCorpus.every(blocks, [replace(task(), manifest=manifest)]).values()
    statics_only = replace(
        corpus,
        manifest=replace(
            corpus.manifest,
            channels=tuple(replace(c, timeless=True) for c in corpus.manifest.channels),
        ),
    )

    with pytest.raises(UnsupportedDetectorFeaturesError, match="measures no channel over time"):
        next(ChannelLocalDetectionRuntime(blocks)._cells(statics_only, [window("a", 0, 10.0)]))


def test_units_whose_windows_interleave_in_time_are_refused_before_a_fit(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = ChannelLocalDetectionRuntime(PublishedCorpusBlocks(store, tmp_path / "workspace"))
    recipe = DetectorRecipe(
        features=DetectorFeatures.CHANNEL_DEPARTURES,
        reading=None,
        neighbours=None,
        contamination=0.1,
        seed=1,
        history=ChannelHistory(span=3),
    )

    # The two units' windows end at the same instants, each on an axis of its own.
    with pytest.raises(OverlappingWindowsError, match="interleave in time"):
        runtime.fit(
            recipe,
            replace(task(), manifest=manifest),
            (window("a", 0, 10.0), window("a", 1, 20.0)),
            (labelled("c", 2, 10.0, 1.0),),
            retain=False,
        )
