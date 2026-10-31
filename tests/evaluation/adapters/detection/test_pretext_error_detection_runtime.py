import io
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.detection.hidden_cells import HiddenCells
from emblema.evaluation.adapters.detection.masked_cells import MaskedCells
from emblema.evaluation.adapters.detection.pretext_error_detection_runtime import (
    PretextErrorDetectionRuntime,
)
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.exceptions import UnreadablePretextError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from tests.evaluation.support import WEIGHTS, labelled, task, window
from tests.support.published import HELD_OUT, publish

FITTING = (window("a", 0, 10.0), window("a", 1, 15.0), window("b", 3, 10.0))
SCORED = (labelled("c", 2, 10.0, 1.0), labelled("a", 0, 10.0, 0.0))
BINS = 2


class FakePretext:
    """Hides every timed cell that held a reading and answers each with its own value plus a miss.

    Every call is recorded with its seeds, so a test can say which windows were asked and how.
    """

    def __init__(
        self,
        missed: Callable[[TokenWindow], float] = lambda _: 0.0,
        *,
        bins: int = BINS,
        huber_delta: float = 0.0,
    ) -> None:
        self._missed = missed
        self._layout = BinnedTokens(max(bins, 1))
        self.bins = bins
        self.huber_delta = huber_delta
        self.calls: list[tuple[list[TokenWindow], list[int], bool]] = []

    def masked(
        self,
        weights: ArtifactRef,
        windows: Sequence[TokenWindow],
        seeds: Sequence[int],
        *,
        answered: bool,
        device: str,
    ) -> list[MaskedCells]:
        self.calls.append((list(windows), list(seeds), answered))
        drawn = []
        for token_window in windows:
            (full,) = self._layout.lay([token_window])
            hidden = (full.features[:, BinnedTokens.PRESENCE] > 0.5) & ~full.timeless
            visible = self._layout.without(token_window, hidden)
            values = full.features[:, BinnedTokens.VALUE_FEATURES]
            drawn.append(
                MaskedCells(
                    full=full.features.astype(np.float64),
                    visible=visible.features.astype(np.float64),
                    hidden=hidden,
                    channel_ids=full.channel_ids.astype(np.int64),
                    timeless=full.timeless,
                    answers=(values + self._missed(token_window)) if answered else None,
                )
            )
        return drawn


def held_out_missed_by_three(token_window: TokenWindow) -> float:
    """A miss of three on the held-out window's cells, read back from the block, and none else."""
    same = len(token_window.values) == len(HELD_OUT.values)
    return 3.0 if same and np.allclose(token_window.values, HELD_OUT.values) else 0.0


def recipe(features: DetectorFeatures = DetectorFeatures.PRETEXT_ERROR) -> DetectorRecipe:
    return DetectorRecipe(
        features=features,
        reading=None,
        neighbours=None,
        contamination=0.1,
        seed=1,
        pretext=PretextAsking(backbone=WEIGHTS, draws=2),
    )


@pytest.fixture
def detection_task(tmp_path: Path) -> tuple[DownstreamTask, PublishedCorpusBlocks]:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    stated = task(
        protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None
    )
    return replace(stated, manifest=manifest), PublishedCorpusBlocks(store, tmp_path / "work")


def test_the_window_the_decoder_misses_most_scores_highest(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    source = FakePretext(held_out_missed_by_three)

    outcome = PretextErrorDetectionRuntime(source, blocks).fit(
        recipe(), stated, FITTING, SCORED, retain=False
    )

    missed, kept = (p.predicted for p in outcome.predictions)
    # Nothing fitted was missed, so the alarm sits at no error and the held-out window above it.
    assert missed == pytest.approx(9.0)
    assert kept == pytest.approx(0.0)


def test_a_miss_past_the_knee_is_read_by_the_backbone_s_bounded_loss(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    source = FakePretext(held_out_missed_by_three, huber_delta=1.0)

    outcome = PretextErrorDetectionRuntime(source, blocks).fit(
        recipe(), stated, FITTING, SCORED, retain=False
    )

    assert outcome.predictions[0].predicted == pytest.approx(1.0 * (3.0 - 0.5))


def test_every_detector_asks_every_window_under_the_same_draws(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    asked: dict[DetectorFeatures, set[tuple[int, int]]] = {}
    for features in (
        DetectorFeatures.PRETEXT_ERROR,
        DetectorFeatures.LINEAR_ANSWER_ERROR,
        DetectorFeatures.INTERPOLATION_ERROR,
    ):
        source = FakePretext()
        PretextErrorDetectionRuntime(source, blocks).fit(
            recipe(features), stated, FITTING, SCORED, retain=False
        )
        asked[features] = {
            (hash(token_window), seed)
            for token_windows, seeds, _ in source.calls
            for token_window, seed in zip(token_windows, seeds, strict=True)
        }

    # Four windows (one fitted and scored), two draws each, the same seeds under all three.
    assert len(asked[DetectorFeatures.PRETEXT_ERROR]) == 4 * 2
    assert asked[DetectorFeatures.PRETEXT_ERROR] == asked[DetectorFeatures.INTERPOLATION_ERROR]
    assert asked[DetectorFeatures.PRETEXT_ERROR] == asked[DetectorFeatures.LINEAR_ANSWER_ERROR]


def test_only_the_backbone_s_own_error_runs_its_decoder(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    for features, decoded in (
        (DetectorFeatures.PRETEXT_ERROR, True),
        (DetectorFeatures.LINEAR_ANSWER_ERROR, False),
        (DetectorFeatures.INTERPOLATION_ERROR, False),
    ):
        source = FakePretext()
        PretextErrorDetectionRuntime(source, blocks).fit(
            recipe(features), stated, FITTING, SCORED, retain=False
        )
        assert {answered for _, _, answered in source.calls} == {decoded}


def test_a_window_with_no_scored_cell_scores_the_fitting_side_s_median_and_is_counted(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
    caplog: pytest.LogCaptureFixture,
) -> None:
    stated, blocks = detection_task

    class HidingNothingOfTheHeldOut(FakePretext):
        def masked(self, *args: object, **kwargs: object) -> list[MaskedCells]:
            drawn = super().masked(*args, **kwargs)  # type: ignore[arg-type]
            return [
                replace(cells, hidden=np.zeros_like(cells.hidden))
                if held_out_missed_by_three(token_window)
                else cells
                for cells, token_window in zip(drawn, args[1], strict=True)  # type: ignore[call-overload]
            ]

    source = HidingNothingOfTheHeldOut(lambda _: 1.0)
    with caplog.at_level(logging.INFO):
        outcome = PretextErrorDetectionRuntime(source, blocks).fit(
            recipe(), stated, FITTING, SCORED, retain=False
        )

    # Every fitted window is missed by one, so its error is one and the alarm sits there; the
    # held-out window, read nowhere, is put at that median rather than at no error at all.
    assert [p.predicted for p in outcome.predictions] == pytest.approx([0.0, 0.0])
    assert "1 of 2 scored windows held no hidden cell" in caplog.text


def test_the_decoder_s_answers_are_read_cell_for_cell_on_the_grid() -> None:
    full = np.zeros((5, 9))
    full[1:, 0] = 1.0
    cells = MaskedCells(
        full=full,
        visible=full,
        hidden=np.asarray([True, True, False, False, True], dtype=np.bool_),
        channel_ids=np.asarray([3, 1, 2, 1, 2], dtype=np.int64),
        timeless=np.asarray([True, False, False, False, False], dtype=np.bool_),
        answers=np.arange(25, dtype=np.float64).reshape(5, 5),
    )

    decoded = PretextErrorDetectionRuntime._decoded(HiddenCells.of(cells, bins=2), cells)

    # The static cell's answer is dropped; then bin by bin, the channels within a bin.
    assert decoded[0, 1].tolist() == list(range(10, 15))
    assert decoded[1, 0].tolist() == list(range(15, 20))


def test_the_ridge_is_fitted_on_the_fitting_side_alone(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    source = FakePretext()

    PretextErrorDetectionRuntime(source, blocks).fit(
        recipe(DetectorFeatures.LINEAR_ANSWER_ERROR), stated, FITTING, SCORED, retain=False
    )

    # The fit, the fitting side's errors, then the scored side's: the first pass holds the
    # fitting windows alone.
    passes = [[hash(token_window) for token_window in windows] for windows, _, _ in source.calls]
    fitted = {hash(token_window) for token_window in source.calls[1][0]}
    assert len(passes) == 3
    assert set(passes[0]) == fitted
    assert not any(held_out_missed_by_three(w) for w in source.calls[0][0])
    assert any(held_out_missed_by_three(w) for w in source.calls[2][0])


def test_what_a_fit_keeps_is_its_alarm_and_the_ridge_with_its_columns(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task
    store = InMemoryArtifactStore()
    runtime = PretextErrorDetectionRuntime(FakePretext(), blocks, store=store)

    outcome = runtime.fit(
        recipe(DetectorFeatures.LINEAR_ANSWER_ERROR), stated, FITTING, SCORED, retain=True
    )

    assert outcome.artifact is not None
    kept = KeptCandidates(store).read(outcome.artifact)
    assert kept.measured.format == PretextErrorDetectionRuntime.FORMAT
    with np.load(io.BytesIO(store.get(kept.measured.artifact))) as archive:
        assert archive["columns"].tolist() == [1, 2]
        assert int(archive["bins"]) == BINS
        for channel in archive["channels"].tolist():
            assert archive[f"weights_{channel}"].shape == (5 + 2 * 2 + BINS, 5)


def test_a_backbone_over_readings_is_refused(
    detection_task: tuple[DownstreamTask, PublishedCorpusBlocks],
) -> None:
    stated, blocks = detection_task

    with pytest.raises(UnreadablePretextError, match="one token per reading"):
        PretextErrorDetectionRuntime(FakePretext(bins=0), blocks).fit(
            recipe(), stated, FITTING, SCORED, retain=False
        )


def test_the_seed_of_a_draw_is_the_window_s_and_the_draw_s_alone() -> None:
    first = PretextErrorDetectionRuntime._seed(1, window("a", 0, 1.0), 0)

    assert first == PretextErrorDetectionRuntime._seed(1, window("a", 0, 99.0), 0)
    assert first != PretextErrorDetectionRuntime._seed(1, window("a", 0, 1.0), 1)
    assert first != PretextErrorDetectionRuntime._seed(2, window("a", 0, 1.0), 0)
    assert 0 <= first < 2**63
