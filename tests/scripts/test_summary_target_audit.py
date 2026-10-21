from pathlib import Path

import numpy as np
import pytest

from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from scripts.pretext_triviality_report import CorpusWindows
from scripts.summary_target_audit import (
    COMBINED,
    INTERPOLATION,
    LEAST_SAMPLES,
    MODEL,
    PARTIAL,
    SCHEDULE,
    STACKED,
    SUMMARIES,
    TOKEN,
    VISIBLE,
    WHOLE,
    RidgeMap,
    Score,
    audited,
    drawn,
    masks_of,
    read_scores,
    render,
    samples_of,
    summarised,
    write_scores,
    write_settings,
)
from tests.support.control_corpus import Control, publish_control

pytestmark = pytest.mark.ml

TOY = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=12)
MASKS = MaskingStrategy(channel_rate=0.3, block_rate=0.6, block_span=0.5, token_rate=0.1)


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> Control:
    return publish_control(tmp_path_factory.mktemp("control"))


def corpus_of(control: Control) -> CorpusWindows:
    windows = control.windows_of(0)
    return CorpusWindows(
        corpus="control-a",
        vocabulary_size=control.vocabulary_size,
        channels_apart=frozenset(),
        training=windows[:8],
        validation=windows[8:16],
        validation_units=tuple(f"unit-{index}" for index in range(8)),
        block_checksum="sha256:0",
        channel_ids=frozenset(range(1, control.vocabulary_size + 1)),
    )


def test_the_summaries_are_the_mean_the_extremes_and_the_slope_over_time() -> None:
    found = summarised(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(found, [3.0, 1.0, 5.0, 2.0])


def test_one_instant_has_no_slope() -> None:
    found = summarised(np.array([4.0, 4.0]), np.array([1.0, 2.0]))
    assert np.isnan(found[3])
    np.testing.assert_allclose(found[:3], [1.5, 1.0, 2.0])


def test_each_sample_summarises_its_channel_s_hidden_values(control: Control) -> None:
    [(batch, masks), *_] = drawn(control.windows_of(0)[:8], MASKS, batch_size=8, seed=1)
    arrays = WindowArrays.of(batch)
    hidden = masks.hidden.numpy() & arrays.observed & ~batch.timeless.numpy()
    samples, tokens = samples_of(batch, masks, {}, frozenset())
    assert samples
    assert len(tokens) == len(samples)
    expected = sum(
        len(np.unique(arrays.channel_ids[row, hidden[row]])) for row in range(arrays.rows)
    )
    assert len(samples) == expected
    parts = {sample.part for sample in samples}
    assert parts <= {PARTIAL, WHOLE}
    for sample in samples:
        assert np.isnan(sample.answers[VISIBLE]).all() == (sample.part == WHOLE)


def test_a_channel_set_apart_yields_no_sample(control: Control) -> None:
    [(batch, masks), *_] = drawn(control.windows_of(0)[:8], MASKS, batch_size=8, seed=1)
    samples, _ = samples_of(batch, masks, {}, frozenset())
    apart = frozenset({samples[0].channel_id})
    kept, _ = samples_of(batch, masks, {}, apart)
    assert all(sample.channel_id not in apart for sample in kept)
    assert len(kept) < len(samples)


def test_an_exact_answer_scores_one_and_the_mean_zero() -> None:
    target = np.array([1.0, 2.0, 4.0])
    assert Score.of(target, target).r2 == pytest.approx(1.0)
    assert Score.of(target, np.full(3, target.mean())).r2 == pytest.approx(0.0)
    assert np.isnan(Score.of(np.ones(3), np.ones(3)).r2)


def test_scores_add_as_the_samples_they_sum() -> None:
    first, second = np.array([1.0, 2.0]), np.array([3.0, 5.0, 8.0])
    answer = np.array([1.5, 2.0]), np.array([3.0, 4.0, 9.0])
    joined = Score.of(np.concatenate([first, second]), np.concatenate(answer))
    added = Score.of(first, answer[0]) + Score.of(second, answer[1])
    assert added.r2 == pytest.approx(joined.r2)


def test_a_ridge_map_recovers_a_line_and_answers_the_mean_below_its_samples() -> None:
    features = np.random.default_rng(0).normal(size=(200, 2))
    target = 3.0 * features[:, 0] - features[:, 1] + 1.0
    fitted = RidgeMap.fitted(features, target)
    np.testing.assert_allclose(fitted.predict(features), target, atol=0.05)
    few = RidgeMap.fitted(features[: LEAST_SAMPLES - 1], target[: LEAST_SAMPLES - 1])
    np.testing.assert_allclose(
        few.predict(features[:3]), np.full(3, target[: LEAST_SAMPLES - 1].mean())
    )


def test_every_answer_is_scored_on_every_summary_and_the_tokens(control: Control) -> None:
    model = MaskedReconstruction(
        SetEncoder.for_vocabulary(TOY, control.vocabulary_size), decoder_layers=1
    )
    rows = audited(corpus_of(control), MASKS, "own", model=model, device="cpu", batch_size=4)
    summaries = {row.summary for row in rows}
    assert summaries == {*SUMMARIES, TOKEN}
    on_summaries = {row.answer for row in rows if row.summary != TOKEN}
    assert on_summaries == {VISIBLE, INTERPOLATION, COMBINED, SCHEDULE, STACKED, MODEL}
    assert {row.answer for row in rows if row.summary == TOKEN} == {
        INTERPOLATION,
        COMBINED,
        MODEL,
    }
    same = {(r.channel_id, r.summary, r.part) for r in rows if r.answer == VISIBLE}
    for answer in (SCHEDULE, STACKED, MODEL):
        counts = {
            (r.channel_id, r.summary, r.part): r.score.count for r in rows if r.answer == answer
        }
        visible = {
            (r.channel_id, r.summary, r.part): r.score.count for r in rows if r.answer == VISIBLE
        }
        assert {key: counts[key] for key in same} == visible


def test_the_same_seeds_score_the_same(control: Control) -> None:
    first = audited(corpus_of(control), MASKS, "own", model=None, device="cpu", batch_size=4)
    second = audited(corpus_of(control), MASKS, "own", model=None, device="cpu", batch_size=4)
    assert first == second


def test_what_is_written_reads_back_and_renders(
    control: Control, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control's few windows give each channel fewer samples than a real corpus's median wants.
    monkeypatch.setattr("scripts.summary_target_audit.LEAST_SCORED", 1)
    rows = audited(corpus_of(control), MASKS, "own", model=None, device="cpu", batch_size=4)
    write_settings(tmp_path, {"backbone": "b", "revision": "r", "device": "cpu"})
    write_scores(tmp_path, rows)
    assert read_scores(tmp_path) == rows
    report = render(tmp_path)
    assert "### control-a, masks own" in report
    assert "| mean | all |" in report


def test_the_masks_of_an_experiment_file_are_its_masking_section() -> None:
    strategy = masks_of(Path("experiments/backbone-mixed4-m-forecast.toml"))
    assert strategy.horizon_rate == 1.0
    assert (strategy.horizon_min_span, strategy.horizon_max_span) == (0.15, 0.5)
    assert strategy.block_rate == 0.0
