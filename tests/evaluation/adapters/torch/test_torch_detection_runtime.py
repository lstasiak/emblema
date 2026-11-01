"""The states runtime asked for the cells themselves: the input the encoder reads, no encoder."""

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.torch.torch_detection_runtime import (  # noqa: E402
    TorchDetectionRuntime,
)
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures  # noqa: E402
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe  # noqa: E402
from emblema.evaluation.domain.detection.score_distance import ScoreDistance  # noqa: E402
from emblema.evaluation.domain.exceptions import UnsupportedDetectorFeaturesError  # noqa: E402
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme  # noqa: E402
from emblema.evaluation.domain.task.downstream_task import DownstreamTask  # noqa: E402
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from tests.evaluation.support import labelled, task, window  # noqa: E402
from tests.support.backbones import SmallBackbones  # noqa: E402
from tests.support.published import CHANNELS, publish  # noqa: E402

pytestmark = pytest.mark.ml

FITTING = (window("a", 0, 10.0), window("a", 1, 15.0), window("b", 3, 10.0))
SCORED = (labelled("c", 2, 10.0, 1.0), labelled("a", 0, 10.0, 0.0))
CELLS = DetectorRecipe(
    features=DetectorFeatures.RAW_CELLS, reading=None, neighbours=2, contamination=0.1, seed=1
)


def runtime(
    tmp_path: Path, *, bins: int
) -> tuple[TorchDetectionRuntime, DownstreamTask, SmallBackbones]:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    stated = replace(
        task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None),
        manifest=manifest,
    )
    backbones = SmallBackbones(vocabulary_size=len(CHANNELS), bins=bins)
    fitted = TorchDetectionRuntime(
        backbones, PublishedCorpusBlocks(store, tmp_path / "workspace"), device="cpu"
    )
    return fitted, stated, backbones


@pytest.mark.parametrize(
    "distance",
    [
        ScoreDistance.NEAREST_NEIGHBOURS,
        ScoreDistance.WHITENED,
        ScoreDistance.CENTRE,
        ScoreDistance.FARTHEST,
        ScoreDistance.SUBSPACE,
    ],
)
def test_the_cells_are_read_under_every_distance_without_an_encoder(
    tmp_path: Path, distance: ScoreDistance
) -> None:
    fitted, stated, backbones = runtime(tmp_path, bins=2)
    recipe = replace(
        CELLS,
        distance=distance,
        neighbours=2 if distance is ScoreDistance.NEAREST_NEIGHBOURS else None,
    )

    outcome = fitted.fit(recipe, stated, FITTING, SCORED, retain=False)

    assert [p.window for p in outcome.predictions] == [s.window for s in SCORED]
    assert all(abs(p.predicted) < float("inf") for p in outcome.predictions)
    # No encoder was built or asked for: the cells went to the detector as they were laid out.
    assert backbones.built == []
    assert backbones.requested == []


def test_the_cells_of_a_backbone_over_readings_are_refused(tmp_path: Path) -> None:
    fitted, stated, _ = runtime(tmp_path, bins=0)

    with pytest.raises(UnsupportedDetectorFeaturesError, match="no cells"):
        fitted.fit(CELLS, stated, FITTING, SCORED, retain=False)
