"""One task, every adapter: what any runtime that fits a detector owes.

The contract says nothing about how good the scores are and everything about what a caller may
rely on: exactly the scored windows answered in the order given, as the distance above the alarm
the detector set; no label read from the fitting side, which arrives as windows alone; a run
without a window on either side refused; and what was fitted kept only where there is somewhere
to keep it.
"""

from collections.abc import Callable
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.in_memory.detection_runtime import InMemoryDetectionRuntime
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    UnsupportedDetectorFeaturesError,
)
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.ports.detection_runtime import DetectionRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.evaluation.support import WEIGHTS, labelled, task, window

# The block of the published test corpus holds four windows; three are fitted on and the
# held-out one is scored.
FITTING = (window("a", 0, 10.0), window("a", 1, 15.0), window("b", 3, 10.0))
SCORED = (labelled("c", 2, 10.0, 1.0),)
# Scored windows of two units, one of which holds two, so each can be read against its unit.
OF_TWO_UNITS = (
    labelled("c", 2, 10.0, 1.0),
    labelled("a", 0, 10.0, 0.0),
    labelled("a", 1, 15.0, 0.0),
)
SUMMARIES = DetectorRecipe(
    features=DetectorFeatures.SUMMARIES, reading=None, neighbours=2, contamination=0.1, seed=1
)
STATES = DetectorRecipe(
    features=DetectorFeatures.ENCODER_STATES,
    reading=EncoderReading(architecture=WEIGHTS, backbone=None),
    neighbours=2,
    contamination=0.1,
    seed=1,
)

# The masks of the stored pretext, which ``over_the_pretext`` puts under this reference.
ASKED = PretextAsking(backbone=WEIGHTS, draws=2)
PRETEXT = DetectorRecipe(
    features=DetectorFeatures.PRETEXT_ERROR,
    reading=None,
    neighbours=None,
    contamination=0.1,
    seed=1,
    pretext=ASKED,
)


class Adapted(NamedTuple):
    """An adapter, its task, where what it fits would be kept, and whose masks it asks under."""

    runtime: DetectionRuntime
    task: DownstreamTask
    store: ArtifactStore
    pretext: ArtifactRef | None = None


def detection_task() -> DownstreamTask:
    return task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None)


def in_memory(tmp_path: Path, *, keeping: bool) -> Adapted:
    store = InMemoryArtifactStore()
    return Adapted(InMemoryDetectionRuntime(store if keeping else None), detection_task(), store)


def over_summaries(tmp_path: Path, *, keeping: bool) -> Adapted:
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.detection.summaries_detection_runtime import (
        SummariesDetectionRuntime,
    )
    from tests.support.published import publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = SummariesDetectionRuntime(
        PublishedCorpusBlocks(store, tmp_path / "workspace"), store=store if keeping else None
    )
    return Adapted(runtime, replace(detection_task(), manifest=manifest), store)


def over_encoder_states(tmp_path: Path, *, keeping: bool) -> Adapted:
    pytest.importorskip("torch")
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.torch.torch_detection_runtime import TorchDetectionRuntime
    from tests.support.backbones import SmallBackbones
    from tests.support.published import CHANNELS, publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = TorchDetectionRuntime(
        SmallBackbones(vocabulary_size=len(CHANNELS)),
        PublishedCorpusBlocks(store, tmp_path / "workspace"),
        device="cpu",
        store=store if keeping else None,
    )
    return Adapted(runtime, replace(detection_task(), manifest=manifest), store)


def over_the_pretext(tmp_path: Path, *, keeping: bool) -> Adapted:
    pytest.importorskip("torch")
    from emblema.entrypoints.restored_backbones import RestoredBackbones
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.detection.pretext_error_detection_runtime import (
        PretextErrorDetectionRuntime,
    )
    from tests.support.pretexts import stored_pretext
    from tests.support.published import publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    pretext = stored_pretext(store)
    runtime = PretextErrorDetectionRuntime(
        RestoredBackbones(store, pretext),
        PublishedCorpusBlocks(store, tmp_path / "workspace"),
        device="cpu",
        store=store if keeping else None,
    )
    return Adapted(runtime, replace(detection_task(), manifest=manifest), store, pretext)


ADAPTERS: list[Callable[..., Adapted]] = [
    in_memory,
    over_summaries,
    over_encoder_states,
    over_the_pretext,
]


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.__name__)
def adapted(request: pytest.FixtureRequest, tmp_path: Path) -> Adapted:
    build: Callable[..., Adapted] = request.param
    return build(tmp_path, keeping=True)


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.__name__)
def keeping_nothing(request: pytest.FixtureRequest, tmp_path: Path) -> Adapted:
    build: Callable[..., Adapted] = request.param
    return build(tmp_path, keeping=False)


def recipe_for(adapted: Adapted) -> DetectorRecipe:
    """The recipe an adapter can fit: an encoder's states where it holds one, else the summaries."""
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        return SUMMARIES
    try:
        from emblema.evaluation.adapters.torch.torch_detection_runtime import (
            TorchDetectionRuntime,
        )
    except ImportError:  # pragma: no cover - the stack is installed wherever torch adapters run
        return SUMMARIES
    if isinstance(adapted.runtime, TorchDetectionRuntime):
        return STATES
    if adapted.pretext is not None:
        return replace(PRETEXT, pretext=replace(ASKED, backbone=adapted.pretext))
    return SUMMARIES


def test_a_detector_read_against_its_units_answers_the_same_windows(adapted: Adapted) -> None:
    recipe = replace(recipe_for(adapted), reference=ScoreReference.OWN_UNIT)

    outcome = adapted.runtime.fit(recipe, adapted.task, FITTING, OF_TWO_UNITS, retain=False)

    assert [p.window for p in outcome.predictions] == [w.window for w in OF_TWO_UNITS]
    assert all(isfinite(p.predicted) for p in outcome.predictions)


def test_a_detector_read_against_its_units_scores_otherwise_than_against_the_fit(
    adapted: Adapted,
) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and scores by the windows' ends alone")
    whole = recipe_for(adapted)

    against_the_fit = adapted.runtime.fit(whole, adapted.task, FITTING, OF_TWO_UNITS, retain=False)
    against_the_unit = adapted.runtime.fit(
        replace(whole, reference=ScoreReference.OWN_UNIT),
        adapted.task,
        FITTING,
        OF_TWO_UNITS,
        retain=False,
    )

    assert [p.predicted for p in against_the_unit.predictions] != [
        p.predicted for p in against_the_fit.predictions
    ]


def test_every_scored_window_is_answered_once_in_the_order_given(adapted: Adapted) -> None:
    outcome = adapted.runtime.fit(recipe_for(adapted), adapted.task, FITTING, SCORED, retain=False)

    assert [p.window for p in outcome.predictions] == [labelled.window for labelled in SCORED]
    assert [p.target for p in outcome.predictions] == [1.0]
    assert outcome.seconds >= 0.0
    assert outcome.artifact is None


def test_a_run_without_a_window_to_fit_on_or_to_answer_is_refused(adapted: Adapted) -> None:
    recipe = recipe_for(adapted)
    with pytest.raises(InvalidScoredOutcomeError, match="fit on"):
        adapted.runtime.fit(recipe, adapted.task, (), SCORED, retain=False)
    with pytest.raises(InvalidScoredOutcomeError, match="answer"):
        adapted.runtime.fit(recipe, adapted.task, FITTING, (), retain=False)


def test_what_was_fitted_is_kept_where_there_is_somewhere_to_keep_it(adapted: Adapted) -> None:
    outcome = adapted.runtime.fit(recipe_for(adapted), adapted.task, FITTING, SCORED, retain=True)

    assert outcome.artifact is not None
    kept = KeptCandidates(adapted.store).read(outcome.artifact)
    assert kept.kind is CandidateKind.DETECTOR
    assert kept.corpus_manifest == adapted.task.manifest


def test_keeping_is_refused_where_there_is_nowhere_to_keep(keeping_nothing: Adapted) -> None:
    with pytest.raises(CandidateNotRetainableError):
        keeping_nothing.runtime.fit(
            recipe_for(keeping_nothing), keeping_nothing.task, FITTING, SCORED, retain=True
        )


def test_a_runtime_refuses_features_it_cannot_produce(adapted: Adapted) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and refuses no features")
    other = STATES if recipe_for(adapted) is SUMMARIES else SUMMARIES

    with pytest.raises(UnsupportedDetectorFeaturesError):
        adapted.runtime.fit(other, adapted.task, FITTING, SCORED, retain=False)


def test_the_counts_of_readings_are_read_where_the_summaries_are_and_nowhere_else(
    adapted: Adapted,
) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and refuses no features")
    counts = replace(SUMMARIES, features=DetectorFeatures.READING_COUNTS)

    if recipe_for(adapted) is SUMMARIES:
        outcome = adapted.runtime.fit(counts, adapted.task, FITTING, SCORED, retain=False)
        assert [p.window for p in outcome.predictions] == [w.window for w in SCORED]
    else:
        with pytest.raises(UnsupportedDetectorFeaturesError):
            adapted.runtime.fit(counts, adapted.task, FITTING, SCORED, retain=False)


def test_the_whitened_distance_answers_the_same_windows_and_scores_them_otherwise(
    adapted: Adapted,
) -> None:
    nearest = recipe_for(adapted)
    if nearest.features.asks_the_pretext:
        pytest.skip("a detector scored by an error has no distance to choose")
    whitened = replace(nearest, distance=ScoreDistance.WHITENED, neighbours=None)

    by_nearest = adapted.runtime.fit(nearest, adapted.task, FITTING, OF_TWO_UNITS, retain=False)
    by_whitened = adapted.runtime.fit(whitened, adapted.task, FITTING, OF_TWO_UNITS, retain=False)

    assert [p.window for p in by_whitened.predictions] == [p.window for p in by_nearest.predictions]
    assert all(isfinite(p.predicted) for p in by_whitened.predictions)
    if not isinstance(adapted.runtime, InMemoryDetectionRuntime):
        assert [p.predicted for p in by_whitened.predictions] != [
            p.predicted for p in by_nearest.predictions
        ]
