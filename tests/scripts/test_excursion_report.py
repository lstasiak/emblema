"""What the excursion report weighs, scores, stores and renders.

The block's values are weighed with known answers; the four readings of one window have known
answers; and a backbone a real run kept is scored on the CPU exactly as the run scored it, so
the reading over every hidden token reproduces the run's own validation loss.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.corpus_saturation_report import PlannedRun, StoredRun, read_run, train
from scripts.excursion_report import (
    BACKBONES,
    BINS,
    CLIPPED,
    HUBER,
    LOSSES,
    MSE,
    ORDINARY,
    SCORES,
    SIDES,
    Concentration,
    Side,
    WindowMass,
    huber,
    main,
    masses_of,
    measure,
    of_loss,
    read_backbones,
    read_bins,
    read_masses,
    read_scores,
    reading_the_same_block,
    readings,
    render,
    restored,
    score,
    sides_of,
    totals,
    write_rows,
    write_settings,
)
from tests.scripts.test_corpus_saturation_report import DEVICE, REVISION, in_memory_tracker
from tests.support.experiments import MIXTURE, budget, configuration, corpus
from tests.support.published import (
    OTHER_TRAINING_UNIT,
    TRAINING_UNIT,
    VALIDATION_UNIT,
    manifest_of,
    publish,
)

pytestmark = pytest.mark.ml

EXPERIMENT = "saturation-test-s"


def published_sides(tmp_path: Path) -> dict[str, Side]:
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "publisher")
    block = WindowBlock(tmp_path / "publisher" / "corpus.block")
    return sides_of(manifest_of(published.block), block)


def test_both_sides_of_the_split_come_out_of_the_block_with_their_units(tmp_path: Path) -> None:
    sides = published_sides(tmp_path)

    assert sides["training"].units == (TRAINING_UNIT, TRAINING_UNIT, OTHER_TRAINING_UNIT)
    assert sides["validation"].units == (VALIDATION_UNIT,)
    assert len(sides["training"].windows) == 3
    assert [window.values for window in sides["training"].every(2).windows] == [
        sides["training"].windows[0].values,
        sides["training"].windows[2].values,
    ]
    assert sides["training"].every(2).units == (TRAINING_UNIT, OTHER_TRAINING_UNIT)


def test_a_side_whose_units_do_not_match_its_windows_is_refused(tmp_path: Path) -> None:
    training = published_sides(tmp_path)["training"]

    with pytest.raises(ValueError, match="units for"):
        Side("training", training.windows, training.units[:-1])


def test_a_window_is_weighed_with_the_tokens_past_the_threshold(tmp_path: Path) -> None:
    training = published_sides(tmp_path)["training"]

    weighed = masses_of(training, threshold=1.0)

    # -0.5, 0.25, 1.5: one token past one deviation, holding its square.
    assert weighed[0] == WindowMass("training", TRAINING_UNIT, 0, 3, 2.5625, 1, 2.25)
    # A timeless token of 2.0 beside 0.125: weighed like any other, as the masks hide any.
    assert weighed[1] == WindowMass("training", TRAINING_UNIT, 1, 2, 4.015625, 1, 4.0)
    # -1 and 1 sit on the threshold and are not past it.
    assert weighed[2] == WindowMass("training", OTHER_TRAINING_UNIT, 2, 2, 2.0, 0, 0.0)


def test_the_concentration_of_a_side_is_read_off_its_windows(tmp_path: Path) -> None:
    weighed = masses_of(published_sides(tmp_path)["training"], threshold=1.0)

    summary = Concentration.of(weighed)

    assert (summary.side, summary.windows, summary.tokens) == ("training", 3, 7)
    assert summary.mean_square == pytest.approx(8.578125 / 7)
    assert summary.past_tokens == 2
    assert summary.past_squares_share == pytest.approx(6.25 / 8.578125)
    assert summary.top_unit == TRAINING_UNIT
    assert summary.top_unit_share == pytest.approx(6.578125 / 8.578125)
    # One per cent of three windows is one window: the heaviest.
    assert summary.top_windows_share == pytest.approx(4.015625 / 8.578125)
    assert summary.ordinary_mean_square == pytest.approx((8.578125 - 6.25) / 5)


def test_rows_of_two_sides_or_of_none_make_no_concentration() -> None:
    of_two = [
        WindowMass("training", "u", 0, 1, 1.0, 0, 0.0),
        WindowMass("validation", "u", 0, 1, 1.0, 0, 0.0),
    ]

    with pytest.raises(ValueError, match="one side"):
        Concentration.of(of_two)
    with pytest.raises(ValueError, match="one side"):
        Concentration.of([])


def test_the_huber_loss_is_quadratic_within_delta_and_linear_past_it() -> None:
    errors = np.asarray([0.0, 0.5, -1.0, 2.0, -15.0])

    assert huber(errors, 1.0).tolist() == [0.0, 0.125, 0.5, 1.5, 14.5]
    assert huber(errors, 3.0).tolist() == [0.0, 0.125, 0.5, 2.0, 40.5]


def test_the_four_readings_of_one_window_have_known_answers() -> None:
    target = np.asarray([0.5, -3.0, 20.0])
    predicted = np.asarray([0.0, -1.0, 5.0])

    read = readings(target, predicted, threshold=10.0, delta=1.0)

    assert read[MSE] == pytest.approx((3, 229.25, 409.25))
    assert read[ORDINARY] == pytest.approx((2, 4.25, 9.25))
    assert read[HUBER] == pytest.approx((3, 16.125, 22.125))
    assert read[CLIPPED] == pytest.approx((3, 29.25, 109.25))


def cpu_runtime(store: ArtifactStore) -> TrainingRuntime:
    return TorchTrainingRuntime(store, device="cpu")


def trained_run(tmp_path: Path, store: ArtifactStore) -> tuple[StoredRun, Side]:
    """A tiny run trained for real on the CPU and stored as the saturation report stores one."""
    windows = corpus(training=8, validation=4)
    stated = configuration(budget=budget(epochs=1, batch_size=2))
    planned = PlannedRun(EXPERIMENT, "invented", stated, windows, 4)
    directory = tmp_path / "runs" / EXPERIMENT / planned.name
    torch.manual_seed(0)
    stored = train(
        planned, directory, cpu_runtime, store, in_memory_tracker, device=DEVICE, revision=REVISION
    )
    return stored, Side("validation", list(windows.validation), ("v1", "v1", "v2", "v2"))


def test_a_stored_backbone_is_scored_exactly_as_its_run_scored_it(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)

    scores, bins = score(run, restored(store, run), validation, MIXTURE, device="cpu")

    observed, hidden, model, trivial = totals(of_loss(scores, MSE))
    last = run.epochs[-1]
    assert model / hidden == pytest.approx(last.validation_loss, rel=1e-5)
    assert hidden / observed == pytest.approx(last.hidden_ratio)
    assert trivial > 0.0
    assert {row.loss for row in scores} == set(LOSSES)
    assert [row.unit for row in of_loss(scores, MSE)] == ["v1", "v1", "v2", "v2"]
    assert [row.window for row in of_loss(scores, MSE)] == [0, 1, 2, 3]
    # Every hidden token falls in one bin, and the bins add up to the reading over every token.
    assert sum(row.tokens for row in bins) == hidden
    assert sum(row.model for row in bins) == pytest.approx(model)
    assert sum(row.trivial for row in bins) == pytest.approx(trivial)
    assert [row.low for row in bins][:3] == [0.0, 1.0, 2.0]


def test_within_the_threshold_leaves_the_excursions_out_and_nothing_else(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    model = restored(store, run)

    wide, _ = score(run, model, validation, MIXTURE, device="cpu", threshold=100.0)
    narrow, _ = score(run, model, validation, MIXTURE, device="cpu", threshold=0.5)

    assert totals(of_loss(wide, ORDINARY)) == totals(of_loss(wide, MSE))
    assert totals(of_loss(narrow, ORDINARY))[1] < totals(of_loss(narrow, MSE))[1]
    assert totals(of_loss(wide, CLIPPED)) == totals(of_loss(wide, MSE))


def test_a_run_that_has_not_ended_kept_no_backbone(tmp_path: Path) -> None:
    directory = tmp_path / "runs" / EXPERIMENT / "1"
    directory.mkdir(parents=True)
    (directory / "run.csv").write_text("key,value\nexperiment,x\n", encoding="utf-8")
    unfinished = read_run(directory)
    assert unfinished is not None

    assert unfinished.backbone is None
    with pytest.raises(ValueError, match="not ended"):
        restored(InMemoryArtifactStore(), unfinished)


def test_the_measurement_scores_every_finished_run_and_stores_what_it_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    directory = tmp_path / "excursions"
    unfinished = StoredRun(tmp_path / "none", dict(run.settings), (), (), None)

    backbones = measure(
        [unfinished, run], store, validation, {EXPERIMENT: MIXTURE}, directory, device="cpu"
    )

    assert [backbone.fraction for backbone in backbones] == [1.0]
    assert backbones[0].run_validation_loss == run.epochs[-1].validation_loss
    assert backbones[0].label == "invented, tier S, 16 wide, 1 deep"
    assert read_backbones(directory) == backbones
    assert len(read_scores(directory)) == 4 * len(LOSSES)
    assert len(read_bins(directory)) == 9
    assert "not finished, not scored" in capsys.readouterr().err


def test_rows_survive_the_round_trip_through_their_files(tmp_path: Path) -> None:
    weighed = [
        WindowMass("validation", "u2", 0, 2, 1 / 3, 1, 0.25),
        WindowMass("validation", "u2", 1, 3, 2.0, 0, 0.0),
    ]

    write_rows(tmp_path / SIDES, weighed)
    write_rows(tmp_path / SCORES, [])

    assert read_masses(tmp_path) == weighed
    assert read_scores(tmp_path) == []
    assert not (tmp_path / SCORES).read_text(encoding="utf-8")


def test_the_report_renders_the_concentration_and_every_table_of_each_experiment(
    tmp_path: Path,
) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    directory = tmp_path / "excursions"
    write_settings(directory, {"corpus": "invented", "threshold": "3", "huber_delta": "2"})
    write_rows(directory / SIDES, masses_of(validation, threshold=3.0))
    measure([run], store, validation, {EXPERIMENT: MIXTURE}, directory, device="cpu")

    report = render(directory, device="cpu", revision=REVISION)

    assert "— excursions" in report
    assert "| Corpus | invented |" in report
    assert "### Where the squared magnitude of each side lies" in report
    assert "| validation | 4 |" in report
    assert f"### {EXPERIMENT}" in report
    assert "Relative, within 3 SD" in report
    assert "Relative, Huber δ=2" in report
    assert f"#### {EXPERIMENT}, by held-out unit" in report
    assert "| `v1` |" in report
    assert f"#### {EXPERIMENT}, by window" in report


def test_the_report_alone_renders_what_is_stored_and_scores_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    measure([run], store, validation, {EXPERIMENT: MIXTURE}, tmp_path / "excursions", device="cpu")

    main(["--report-only", "--workspace", str(tmp_path), "--device", "cpu"])

    out = capsys.readouterr().out
    assert f"### {EXPERIMENT}" in out
    assert (tmp_path / "excursions" / BACKBONES).is_file()
    assert (tmp_path / "excursions" / BINS).is_file()


def test_scoring_needs_a_corpus_to_score_against(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--workspace", str(tmp_path), "--device", "cpu"])


def test_a_window_of_the_block_reads_back_as_a_token_window(tmp_path: Path) -> None:
    validation = published_sides(tmp_path)["validation"]

    assert isinstance(validation.windows[0], TokenWindow)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="scores on the accelerator")
def test_a_backbone_scores_the_same_on_the_accelerator_as_on_the_host(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    model = restored(store, run)

    on_host, _ = score(run, model, validation, MIXTURE, device="cpu")
    on_device, _ = score(run, model, validation, MIXTURE, device="mps")

    for loss in LOSSES:
        _, tokens, model_sum, trivial = totals(of_loss(on_host, loss))
        _, device_tokens, device_model, device_trivial = totals(of_loss(on_device, loss))
        assert device_tokens == tokens
        assert device_trivial == pytest.approx(trivial)
        assert device_model == pytest.approx(model_sum, rel=1e-3)


def test_backbones_are_not_scored_against_a_block_their_runs_never_read(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    run, _ = trained_run(tmp_path, store)
    other = ArtifactRef("durable/other-block", Checksum.of_bytes(b"another publication"))

    reading_the_same_block([run], ArtifactRef("b", Checksum.parse(run.settings["block_checksum"])))
    with pytest.raises(ValueError, match="another corpus"):
        reading_the_same_block([run], other)


def test_scoring_under_other_masks_than_the_run_s_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    other = replace(MIXTURE, token_rate=0.9)

    measure([run], store, validation, {EXPERIMENT: other}, tmp_path / "excursions", device="cpu")

    assert "not the ones it was scored on" in capsys.readouterr().err
