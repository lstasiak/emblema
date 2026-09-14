"""The report's verdicts are code, so they are checked from both sides here.

What the training measures depends on the machine; what the report says about a measurement does
not. A report that could only ever print the reassuring answer would be worse than none, so each
verdict is exercised on numbers that should earn it and on numbers that should not. The whole
road — publish, train, diagnose, assess, store, draw — runs once on a few units with a toy encoder.
"""

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.mask_kind import MaskKind
from scripts.masked_reconstruction_assessment import INDEX, find_shorter, read, store
from scripts.masked_reconstruction_report import (
    MIXTURE,
    Published,
    Run,
    Trained,
    code_digest,
    diagnose,
    draw_figures,
    evaluate,
    main,
    measured_epochs,
    pick_examples,
    publish,
    render,
    results_of,
    shape_of,
    spectrum_verdict,
    train,
    verdict_section,
)
from tests.support.token_tensors import grid_batch

pytestmark = pytest.mark.ml

assess = AssessReconstructionRun(UnitBootstrap())

TOY = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=12)


def run_named(corpus: str, *, epochs: int = 2) -> Run:
    return Run(
        corpus=corpus,
        tier="S",
        shape=TOY,
        units=4,
        epochs=epochs,
        batch_size=16,
        learning_rate=1e-3,
        warmup_epochs=1,
        final_lr_fraction=0.01,
        seed=1,
        device="cpu",
    )


def tallies(kind: MaskKind, *, model: float, matched: float) -> list[MaskKindTally]:
    return [
        MaskKindTally(
            kind, False, f"u/{unit}", 100, model * 100, matched * 100, matched * 100, 100.0, 1.0
        )
        for unit in range(12)
    ]


def results(
    *kinds: list[MaskKindTally],
    validation: tuple[float, ...] = (0.6, 0.5),
    truth: tuple[float, ...] = (1.0, 1.0),
    residual: tuple[float, ...] = (0.0, 1.0),
) -> Results:
    return Results(
        settings={"corpus": "control-a", "date": "2026-09-13 10:00:00"},
        strategy=MIXTURE,
        realised_ratio=0.46,
        curve=Curve(validation, validation, (1.0,) * len(validation)),
        tallies=tuple(tally for kind in kinds for tally in kind),
        spectrum=Spectrum(truth, residual, residual, fitted=4, skipped=0),
    )


def every_kind_learnt() -> list[list[MaskKindTally]]:
    return [
        tallies(MaskKind.CHANNEL, model=0.1, matched=0.3),
        tallies(MaskKind.BLOCK, model=0.1, matched=0.2),
        tallies(MaskKind.TOKEN, model=0.01, matched=0.02),
    ]


def test_the_shape_flag_keeps_the_tiers_time_frequencies() -> None:
    shape = shape_of("64,2,2,256")

    assert (shape.width, shape.heads, shape.layers, shape.feedforward_width) == (64, 2, 2, 256)
    assert shape.time_frequencies == 12


def test_a_run_without_a_shape_builds_the_tier() -> None:
    run = replace(run_named("control-a"), shape=None)

    assert run.architecture.width == 192
    assert run.shape_label == "tier S"
    assert run_named("control-a").shape_label == "tier S cut down to 16 wide, 1 deep"


def test_the_schedule_counts_its_warmup_and_its_length_in_optimiser_steps() -> None:
    schedule = run_named("control-a", epochs=5).schedule(steps_per_epoch=10)

    assert (schedule.warmup_steps, schedule.total_steps) == (10, 50)
    assert schedule.final_fraction == 0.01


def test_the_verdict_says_when_the_loss_did_not_fall() -> None:
    run = run_named("control-a")
    rose = verdict_section(run, assess(results(*every_kind_learnt(), validation=(0.5, 0.6))))
    fell = verdict_section(run, assess(results(*every_kind_learnt())))

    assert "did not fall" in rose
    assert "Loss fell" in fell


def test_the_verdict_reads_each_kind_off_its_interval_and_names_the_trivial_ones() -> None:
    losing = [*every_kind_learnt()[:2], tallies(MaskKind.TOKEN, model=0.02, matched=0.01)]

    text = verdict_section(run_named("control-a"), assess(results(*losing)))

    token = text.split("`token` masks")[1].split("\n")[0]
    block = text.split("`block` masks")[1].split("\n")[0]
    assert "— beaten" in token
    assert "against the linear baseline" in token
    assert "— learnt" in block
    assert "**positive**" in text


def test_the_verdict_is_negative_only_when_every_drawn_kind_was_learnt() -> None:
    learnt = verdict_section(run_named("control-a"), assess(results(*every_kind_learnt())))
    missing = verdict_section(run_named("control-a"), assess(results(*every_kind_learnt()[:2])))

    assert "**negative**" in learnt
    assert "**positive**" in missing


def test_a_spectrum_with_signal_at_one_frequency_is_called_uninformative() -> None:
    blind = spectrum_verdict(Spectrum((1000.0, 3.0, 1.0), (1.0, 9.0, 9.0), (1.0,) * 3, 4, 0))

    assert "uninformative" in blind
    assert "recovered" not in blind.split("—")[0]


def test_a_spectrum_with_signal_at_several_frequencies_says_where_it_is_lost() -> None:
    spread = Spectrum((100.0, 0.1, 50.0), (10.0, 0.08, 35.0), (1.0,) * 3, 4, 0)
    everywhere = Spectrum((100.0, 50.0), (10.0, 5.0), (1.0, 1.0), 4, 0)

    assert "energy at 1, 3 cycle(s)" in spectrum_verdict(spread)
    assert "90% at 1, 30% at 3" in spectrum_verdict(spread)
    assert "loses the signal at 3" in spectrum_verdict(spread)
    assert "every frequency the signal has is recovered" in spectrum_verdict(everywhere)
    assert "no channel hidden whole" in spectrum_verdict(replace(everywhere, fitted=0))


def test_an_example_draws_only_the_tokens_its_kind_hid() -> None:
    batch = grid_batch(1, 2, 11, seed=1)
    ids, times = batch.channel_ids, batch.timestamps
    # Channel 1 loses a block in the middle and, separately, its first token.
    block = (ids == 1) & (times > 0.3) & (times < 0.7)
    token = (ids == 1) & (times == 0.0)
    masks = TokenMasks(channel=torch.zeros_like(block), block=block, token=token)
    prediction = torch.zeros_like(batch.features[..., 0])

    examples = pick_examples(
        batch,
        masks,
        prediction,
        prediction,
        prediction,
        first_window=7,
        channel_name=lambda channel: f"s{channel}",
        apart=frozenset({3}),
    )

    by_kind = {example.kind: example for example in examples}
    assert set(by_kind) == {MaskKind.BLOCK, MaskKind.TOKEN}
    assert by_kind[MaskKind.BLOCK].of_kind.sum() == 3
    assert by_kind[MaskKind.TOKEN].of_kind.sum() == 1
    assert not (by_kind[MaskKind.BLOCK].of_kind & by_kind[MaskKind.BLOCK].visible).any()
    assert by_kind[MaskKind.BLOCK].visible.sum() == 11 - 3 - 1
    assert {example.window for example in examples} == {7}


def test_measured_epochs_are_read_per_tier_and_corpus(tmp_path: Path) -> None:
    epochs = tmp_path / "epochs.toml"
    epochs.write_text("[S]\ncontrol-a = 24\n\n[M]\n", encoding="utf-8")

    assert measured_epochs("S", "control-a", path=epochs) == 24
    assert measured_epochs("S", "control-b", path=epochs) is None
    assert measured_epochs("M", "control-a", path=epochs) is None
    assert measured_epochs("L", "control-a", path=epochs) is None


@pytest.mark.parametrize("stated", ["0", "-3", '"24"', "true", "2.5"])
def test_measured_epochs_that_are_not_a_positive_whole_number_are_refused(
    tmp_path: Path, stated: str
) -> None:
    epochs = tmp_path / "epochs.toml"
    epochs.write_text(f"[S]\ncontrol-a = {stated}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be positive"):
        measured_epochs("S", "control-a", path=epochs)


def test_a_corpus_without_measured_epochs_is_refused_before_anything_runs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--corpus", "null-a", "--workspace", str(tmp_path)])
    assert not any(tmp_path.iterdir())


def test_a_device_the_machine_lacks_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(SystemExit):
        main(
            [
                "--corpus",
                "control-a",
                "--epochs",
                "2",
                "--device",
                "mps",
                "--workspace",
                str(tmp_path),
            ]
        )
    assert "device mps is not available" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_a_schedule_that_leaves_nothing_to_decay_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["--corpus", "control-a", "--epochs", "1", "--workspace", str(tmp_path)])
    assert "warmup_steps" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_the_code_digest_is_stable() -> None:
    digest = code_digest()

    assert digest == code_digest()
    assert len(digest) == 16
    int(digest, 16)


@dataclass(frozen=True)
class Road:
    """Everything one toy run leaves behind, for the tests that read it."""

    run: Run
    published: Published
    trained: Trained
    report: str
    figures: Path
    stored: Path


@pytest.fixture(scope="module")
def road(tmp_path_factory: pytest.TempPathFactory) -> Road:
    run = run_named("control-a")
    workspace = tmp_path_factory.mktemp("report")
    published = publish(run, workspace / "corpus")
    trained = train(published, run)
    found = diagnose(published, trained, run)
    measured = results_of(run, published, trained, found)
    assessment = assess(measured, find_shorter(measured, workspace / "results"))
    draw_figures(run, trained, found, assessment, workspace / "figures")
    stored = store(assessment, workspace / "results")
    report = render(run, published, trained, found, assessment)
    return Road(run, published, trained, report, workspace / "figures", stored)


def test_the_whole_road_renders_every_section(road: Road) -> None:
    for section in ("### Loss", "### Triviality", "### Spectral", "### Verdict", "### Assessment"):
        assert section in road.report
    assert "control-a" in road.report
    assert "cut to 4 units" in road.report
    assert "1 epoch(s) of warmup then cosine decay to 0.01 of the peak, seed 1, CPU" in road.report


def test_the_whole_road_draws_three_figures(road: Road) -> None:
    assert sorted(path.name for path in road.figures.iterdir()) == [
        "masked-reconstruction-control-a-diagnostics.png",
        "masked-reconstruction-control-a-loss.png",
        "masked-reconstruction-control-a-windows.png",
    ]


def test_the_whole_road_stores_what_it_measured_per_unit(road: Road) -> None:
    measured = read(road.stored)

    assert (road.stored.parent / INDEX).is_file()
    assert measured.epochs == road.run.epochs
    assert measured.strategy == MIXTURE
    assert measured.settings["device"] == "cpu"
    assert measured.settings["time_frequencies"] == str(TOY.time_frequencies)
    assert measured.settings["validation_units"] == str(len(set(road.published.validation_units)))
    assert {tally.group for tally in measured.tallies} <= set(road.published.validation_units)
    assert all(tally.floor is not None for tally in measured.tallies)
    # Noise is what no predictor recovers, so a toy model after two epochs stays above it.
    assert all(tally.model >= (tally.floor or 0.0) for tally in measured.tallies if not tally.apart)


def test_the_noise_variance_is_the_layouts_noise_in_normalised_units(road: Road) -> None:
    published = road.published
    variance = published.noise_variance()
    scheme = published.manifest.scheme

    assert variance is not None
    for entry in scheme.vocabulary.entries_of(published.manifest.corpus):
        statistics = scheme.statistics[entry.channel_id - 1]
        if entry.channel_id in published.channels_apart():
            assert variance[entry.channel_id] == 0.0
        else:
            assert statistics is not None
            assert variance[entry.channel_id] == pytest.approx(
                (published.noise or 0.0) ** 2 / statistics.std**2
            )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_a_model_on_mps_is_scored_under_the_same_masks_as_on_the_cpu(road: Road) -> None:
    on_mps = copy.deepcopy(road.trained.model).to("mps")

    cpu_loss, cpu_ratio = evaluate(road.trained.model, road.published, road.run)
    mps_loss, mps_ratio = evaluate(on_mps, road.published, replace(road.run, device="mps"))

    assert mps_ratio == cpu_ratio
    assert mps_loss == pytest.approx(cpu_loss, rel=1e-4)


def test_the_sparse_layout_fits_fewer_cycles_than_the_dense_one(tmp_path: Path) -> None:
    dense = publish(run_named("control-a"), tmp_path / "a")
    sparse = publish(run_named("control-b"), tmp_path / "b")

    assert dense.spectral_cycles() == 6
    assert 1 <= sparse.spectral_cycles() < dense.spectral_cycles()
