"""The report's verdicts are code, so they are checked from both sides here.

What the training measures depends on the machine; what the report says about a measurement does
not. A report that could only ever print the reassuring answer would be worse than none, so each
verdict is exercised on numbers that should earn it and on numbers that should not. The whole
road — publish, train, diagnose, draw — runs once on a few units with a toy encoder.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.pretraining.adapters.diagnostics.mask_kind_verdict import MaskKindVerdict
from emblema.pretraining.adapters.diagnostics.spectral_recovery import (
    SpectralRecovery,
)
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.mask_kind import MaskKind
from scripts.masked_reconstruction_report import (
    Diagnosis,
    Epoch,
    Run,
    Trained,
    diagnose,
    draw_figures,
    informative_frequencies,
    publish,
    render,
    shape_of,
    train,
    verdict_section,
)

pytestmark = pytest.mark.ml

TOY = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=12)


def run_named(corpus: str) -> Run:
    return Run(
        corpus=corpus,
        tier="S",
        shape=TOY,
        units=4,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        seed=1,
    )


def verdict(
    kind: MaskKind, *, model: float, baseline: float, apart: bool = False
) -> MaskKindVerdict:
    return MaskKindVerdict(
        kind=kind, apart=apart, tokens=100, model_error=model, baseline_error=baseline
    )


def diagnosis(*verdicts: MaskKindVerdict, recovered: tuple[float, ...] = (1.0, 0.0)) -> Diagnosis:
    truth = (1.0,) * len(recovered)
    residual = tuple(1.0 - share for share in recovered)
    spectrum = SpectralRecovery(len(recovered), truth, residual, fitted=4, skipped=0)
    return Diagnosis(
        verdicts=verdicts,
        interpolation_loss=0.2,
        ridge_loss=0.3,
        spectrum_of_model=spectrum,
        spectrum_of_ridge=spectrum,
        examples=(),
    )


def trained(*validation: float) -> Trained:
    epochs = tuple(Epoch(training_loss=v + 0.1, validation_loss=v, seconds=1.0) for v in validation)
    return Trained(model=None, epochs=epochs, hidden_ratio=0.45)  # type: ignore[arg-type]


def test_the_shape_flag_keeps_the_tiers_time_frequencies() -> None:
    shape = shape_of("64,2,2,256")

    assert (shape.width, shape.heads, shape.layers, shape.feedforward_width) == (64, 2, 2, 256)
    assert shape.time_frequencies == 12


def test_a_run_without_a_shape_builds_the_tier() -> None:
    run = Run(
        corpus="control-a",
        tier="S",
        shape=None,
        units=None,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        seed=1,
    )

    assert run.architecture.width == 192
    assert run.shape_label == "tier S"
    assert run_named("control-a").shape_label == "tier S cut down to 16 wide, 1 deep"


def test_the_diagnostic_is_negative_only_when_every_kind_beats_its_baseline() -> None:
    beaten = diagnosis(
        verdict(MaskKind.CHANNEL, model=0.1, baseline=0.3),
        verdict(MaskKind.BLOCK, model=0.1, baseline=0.2),
    )
    matched = diagnosis(
        verdict(MaskKind.CHANNEL, model=0.1, baseline=0.3),
        verdict(MaskKind.BLOCK, model=0.2, baseline=0.2),
    )

    assert beaten.negative
    assert not matched.negative


def test_channels_reported_apart_do_not_decide_the_verdict() -> None:
    aside = diagnosis(
        verdict(MaskKind.CHANNEL, model=0.1, baseline=0.3),
        verdict(MaskKind.CHANNEL, model=0.9, baseline=0.3, apart=True),
    )

    assert aside.negative
    assert "positive" not in verdict_section(run_named("control-a"), trained(1.0, 0.5), aside)


def test_the_verdict_says_when_the_loss_did_not_fall() -> None:
    rose = verdict_section(run_named("control-a"), trained(0.5, 0.6), diagnosis())
    fell = verdict_section(run_named("control-a"), trained(0.6, 0.5), diagnosis())

    assert "did not fall" in rose
    assert "Loss fell" in fell


def test_the_verdict_names_the_trivial_kinds() -> None:
    text = verdict_section(
        run_named("control-a"),
        trained(0.6, 0.5),
        diagnosis(
            verdict(MaskKind.TOKEN, model=0.02, baseline=0.01),
            verdict(MaskKind.BLOCK, model=0.1, baseline=0.2),
        ),
    )

    assert "`token` masks" in text
    assert "trivial" in text.split("`token` masks")[1].split("\n")[0]
    assert "learnt" in text.split("`block` masks")[1].split("\n")[0]
    assert "**positive**" in text


def test_the_verdict_reads_the_spectrum_where_the_signal_is() -> None:
    run = run_named("control-a")
    # Energy at cycles 1 and 3 only; the model recovers the first and loses the second.
    found = diagnosis(recovered=(0.9, 0.2, 0.3))
    concentrated = Diagnosis(
        verdicts=(),
        interpolation_loss=0.2,
        ridge_loss=0.3,
        spectrum_of_model=SpectralRecovery(3, (100.0, 0.1, 50.0), (10.0, 0.08, 35.0), 4, 0),
        spectrum_of_ridge=found.spectrum_of_ridge,
        examples=(),
    )

    text = verdict_section(run, trained(0.6, 0.5), concentrated)

    assert "energy at 1, 3 cycle(s)" in text
    assert "90% at 1, 30% at 3" in text
    assert "loses the signal at 3" in text
    assert informative_frequencies(concentrated.spectrum_of_model) == [1, 3]


def test_a_spectrum_recovered_everywhere_says_so() -> None:
    text = verdict_section(
        run_named("control-a"), trained(0.6, 0.5), diagnosis(recovered=(0.9, 0.8))
    )

    assert "every frequency the signal has is recovered" in text


@pytest.fixture(scope="module")
def road(tmp_path_factory: pytest.TempPathFactory) -> tuple[Run, str, Path]:
    run = run_named("control-a")
    workspace = tmp_path_factory.mktemp("report")
    published = publish(run, workspace / "corpus")
    fitted = train(published, run)
    found = diagnose(published, fitted, run)
    draw_figures(run, fitted, found, workspace / "figures")
    return run, render(run, published, fitted, found), workspace / "figures"


def test_the_whole_road_renders_every_section(road: tuple[Run, str, Path]) -> None:
    _, report, _ = road

    for section in ("### Loss", "### Triviality", "### Spectral", "### Verdict"):
        assert section in report
    assert "control-a" in report
    assert "cut to 4 units" in report


def test_the_whole_road_draws_three_figures(road: tuple[Run, str, Path]) -> None:
    _, _, figures = road

    assert sorted(path.name for path in figures.iterdir()) == [
        "masked-reconstruction-control-a-diagnostics.png",
        "masked-reconstruction-control-a-loss.png",
        "masked-reconstruction-control-a-windows.png",
    ]


def test_the_sparse_layout_fits_fewer_cycles_than_the_dense_one(
    tmp_path: Path,
) -> None:
    dense = publish(run_named("control-a"), tmp_path / "a")
    sparse = publish(run_named("control-b"), tmp_path / "b")

    assert dense.spectral_cycles() == 6
    assert 1 <= sparse.spectral_cycles() < dense.spectral_cycles()
