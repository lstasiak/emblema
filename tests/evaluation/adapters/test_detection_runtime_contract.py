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
from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.detection.taught_head import TaughtHead
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    InvalidTaughtEventsError,
    UnsupportedDetectorFeaturesError,
)
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
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


# The same windows on one time axis, one unit after another, for a detector whose history
# runs across the units; the block's positions are the ones above.
ON_ONE_AXIS = (
    (window("a", 0, 10.0), window("a", 1, 20.0), window("b", 3, 30.0)),
    (labelled("c", 2, 40.0, 1.0),),
    (labelled("c", 2, 40.0, 1.0), labelled("a", 0, 10.0, 0.0), labelled("a", 1, 20.0, 0.0)),
)
# A history of two windows back, one of which a channel needs, so every window with an earlier
# one departs from it.
HISTORY = DetectorRecipe(
    features=DetectorFeatures.CHANNEL_DEPARTURES,
    reading=None,
    neighbours=None,
    contamination=0.1,
    seed=1,
    history=ChannelHistory(span=3),
)


class Adapted(NamedTuple):
    """An adapter, its task, where what it fits would be kept, whose masks it asks under, and
    the windows it is handed."""

    runtime: DetectionRuntime
    task: DownstreamTask
    store: ArtifactStore
    pretext: ArtifactRef | None = None
    fitting: tuple[TaskWindow, ...] = FITTING
    scored: tuple[LabelledWindow, ...] = SCORED
    of_two_units: tuple[LabelledWindow, ...] = OF_TWO_UNITS


def learns_no_event(recipe: DetectorRecipe) -> bool:
    return recipe.features.asks_the_pretext or recipe.features.reads_channel_history


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


def over_channel_history(tmp_path: Path, *, keeping: bool) -> Adapted:
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.detection.channel_local_detection_runtime import (
        ChannelLocalDetectionRuntime,
    )
    from tests.support.published import publish

    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = ChannelLocalDetectionRuntime(
        PublishedCorpusBlocks(store, tmp_path / "workspace"), store=store if keeping else None
    )
    fitting, scored, of_two_units = ON_ONE_AXIS
    return Adapted(
        runtime,
        replace(detection_task(), manifest=manifest),
        store,
        fitting=fitting,
        scored=scored,
        of_two_units=of_two_units,
    )


ADAPTERS: list[Callable[..., Adapted]] = [
    in_memory,
    over_summaries,
    over_encoder_states,
    over_the_pretext,
    over_channel_history,
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
    from emblema.evaluation.adapters.detection.channel_local_detection_runtime import (
        ChannelLocalDetectionRuntime,
    )

    if isinstance(adapted.runtime, ChannelLocalDetectionRuntime):
        return HISTORY
    return SUMMARIES


def test_a_detector_read_against_its_units_answers_the_same_windows(adapted: Adapted) -> None:
    recipe = replace(recipe_for(adapted), reference=ScoreReference.OWN_UNIT)

    outcome = adapted.runtime.fit(
        recipe, adapted.task, adapted.fitting, adapted.of_two_units, retain=False
    )

    assert [p.window for p in outcome.predictions] == [w.window for w in adapted.of_two_units]
    assert all(isfinite(p.predicted) for p in outcome.predictions)


def test_a_detector_read_against_its_units_scores_otherwise_than_against_the_fit(
    adapted: Adapted,
) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and scores by the windows' ends alone")
    whole = recipe_for(adapted)

    against_the_fit = adapted.runtime.fit(
        whole, adapted.task, adapted.fitting, adapted.of_two_units, retain=False
    )
    against_the_unit = adapted.runtime.fit(
        replace(whole, reference=ScoreReference.OWN_UNIT),
        adapted.task,
        adapted.fitting,
        adapted.of_two_units,
        retain=False,
    )

    assert [p.predicted for p in against_the_unit.predictions] != [
        p.predicted for p in against_the_fit.predictions
    ]


def test_every_scored_window_is_answered_once_in_the_order_given(adapted: Adapted) -> None:
    outcome = adapted.runtime.fit(
        recipe_for(adapted), adapted.task, adapted.fitting, adapted.scored, retain=False
    )

    assert [p.window for p in outcome.predictions] == [
        labelled.window for labelled in adapted.scored
    ]
    assert [p.target for p in outcome.predictions] == [1.0]
    assert outcome.seconds >= 0.0
    assert outcome.artifact is None


def test_a_run_without_a_window_to_fit_on_or_to_answer_is_refused(adapted: Adapted) -> None:
    recipe = recipe_for(adapted)
    with pytest.raises(InvalidScoredOutcomeError, match="fit on"):
        adapted.runtime.fit(recipe, adapted.task, (), adapted.scored, retain=False)
    with pytest.raises(InvalidScoredOutcomeError, match="answer"):
        adapted.runtime.fit(recipe, adapted.task, adapted.fitting, (), retain=False)


def test_what_was_fitted_is_kept_where_there_is_somewhere_to_keep_it(adapted: Adapted) -> None:
    outcome = adapted.runtime.fit(
        recipe_for(adapted), adapted.task, adapted.fitting, adapted.scored, retain=True
    )

    assert outcome.artifact is not None
    kept = KeptCandidates(adapted.store).read(outcome.artifact)
    assert kept.kind is CandidateKind.DETECTOR
    assert kept.corpus_manifest == adapted.task.manifest


def test_keeping_is_refused_where_there_is_nowhere_to_keep(keeping_nothing: Adapted) -> None:
    with pytest.raises(CandidateNotRetainableError):
        keeping_nothing.runtime.fit(
            recipe_for(keeping_nothing),
            keeping_nothing.task,
            keeping_nothing.fitting,
            keeping_nothing.scored,
            retain=True,
        )


def test_a_runtime_refuses_features_it_cannot_produce(adapted: Adapted) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and refuses no features")
    other = STATES if recipe_for(adapted) is SUMMARIES else SUMMARIES

    with pytest.raises(UnsupportedDetectorFeaturesError):
        adapted.runtime.fit(other, adapted.task, adapted.fitting, adapted.scored, retain=False)


def test_the_counts_of_readings_are_read_where_the_summaries_are_and_nowhere_else(
    adapted: Adapted,
) -> None:
    if isinstance(adapted.runtime, InMemoryDetectionRuntime):
        pytest.skip("the fake reads nothing and refuses no features")
    counts = replace(SUMMARIES, features=DetectorFeatures.READING_COUNTS)

    if recipe_for(adapted) is SUMMARIES:
        outcome = adapted.runtime.fit(
            counts, adapted.task, adapted.fitting, adapted.scored, retain=False
        )
        assert [p.window for p in outcome.predictions] == [w.window for w in adapted.scored]
    else:
        with pytest.raises(UnsupportedDetectorFeaturesError):
            adapted.runtime.fit(counts, adapted.task, adapted.fitting, adapted.scored, retain=False)


def test_the_whitened_distance_answers_the_same_windows_and_scores_them_otherwise(
    adapted: Adapted,
) -> None:
    nearest = recipe_for(adapted)
    if nearest.features.asks_the_pretext or nearest.features.reads_channel_history:
        pytest.skip("a detector scored by an error or by its history has no distance to choose")
    whitened = replace(nearest, distance=ScoreDistance.WHITENED, neighbours=None)

    by_nearest = adapted.runtime.fit(
        nearest, adapted.task, adapted.fitting, adapted.of_two_units, retain=False
    )
    by_whitened = adapted.runtime.fit(
        whitened, adapted.task, adapted.fitting, adapted.of_two_units, retain=False
    )

    assert [p.window for p in by_whitened.predictions] == [p.window for p in by_nearest.predictions]
    assert all(isfinite(p.predicted) for p in by_whitened.predictions)
    if not isinstance(adapted.runtime, InMemoryDetectionRuntime):
        assert [p.predicted for p in by_whitened.predictions] != [
            p.predicted for p in by_nearest.predictions
        ]


# One fitting window taught as an event, which leaves two fitting windows outside it.
TAUGHT = (labelled("a", 1, 15.0, 7.0),)


@pytest.mark.parametrize("head", [TaughtHead.DIRECTION, TaughtHead.COMBINED, TaughtHead.TREES])
def test_a_detector_taught_an_event_answers_the_same_windows_under_every_head(
    adapted: Adapted, head: TaughtHead
) -> None:
    plain = recipe_for(adapted)
    if learns_no_event(plain):
        with pytest.raises(InvalidTaughtEventsError, match="learns from no event"):
            adapted.runtime.fit(
                plain, adapted.task, adapted.fitting, adapted.scored, taught=TAUGHT, retain=False
            )
        return
    taught = replace(plain, events=1, neighbours=None, head=head)

    outcome = adapted.runtime.fit(
        taught, adapted.task, adapted.fitting, adapted.of_two_units, taught=TAUGHT, retain=True
    )

    assert [p.window for p in outcome.predictions] == [w.window for w in adapted.of_two_units]
    assert all(isfinite(p.predicted) for p in outcome.predictions)
    assert outcome.artifact is not None


def test_a_detector_taught_an_event_is_refused_a_window_off_the_fitting_side(
    adapted: Adapted,
) -> None:
    plain = recipe_for(adapted)
    if learns_no_event(plain):
        pytest.skip("a detector scored by an error or by its history learns from no event")
    taught = replace(plain, events=1, neighbours=None)

    with pytest.raises(InvalidTaughtEventsError, match="not on the fitting side"):
        adapted.runtime.fit(
            taught,
            adapted.task,
            adapted.fitting,
            adapted.scored,
            taught=(labelled("c", 2, 10.0, 1.0),),
            retain=False,
        )


def test_a_detector_is_handed_the_events_its_recipe_is_taught_and_no_others(
    adapted: Adapted,
) -> None:
    plain = recipe_for(adapted)
    if learns_no_event(plain):
        pytest.skip("a detector scored by an error or by its history learns from no event")
    taught = replace(plain, events=2, neighbours=None)

    with pytest.raises(InvalidTaughtEventsError, match="was handed 1"):
        adapted.runtime.fit(
            taught, adapted.task, adapted.fitting, adapted.scored, taught=TAUGHT, retain=False
        )
    with pytest.raises(InvalidTaughtEventsError, match="handed no taught window"):
        adapted.runtime.fit(taught, adapted.task, adapted.fitting, adapted.scored, retain=False)
    with pytest.raises(InvalidTaughtEventsError, match="taught no event"):
        adapted.runtime.fit(
            plain, adapted.task, adapted.fitting, adapted.scored, taught=TAUGHT, retain=False
        )
