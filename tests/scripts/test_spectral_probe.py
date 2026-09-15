"""The spectral probe exists to make the spectrum answerable, so that is what is held to here.

It must be the dense control with one dial turned and nothing else, stay out of the control's
registry, keep every harmonic inside what the report fits and samples well, and — the claim itself
— spread the energy of a hidden channel over several frequencies where the control holds it at one.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_PROCESS, LAYOUTS
from emblema.entrypoints.cli.known_corpora import KnownCorpora
from emblema.pretraining.adapters.diagnostics.spectral_recovery import SpectralRecovery
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from scripts.masked_reconstruction_report import (
    CYCLES_AT_MOST,
    WINDOW,
    Run,
    corpora,
    experiments,
    generated,
    publish,
    validation_batches,
)
from scripts.spectral_probe import SPECTRAL_PROBE, SPECTRAL_PROCESS
from tests.support.experiments import budget, configuration

pytestmark = pytest.mark.ml


def run_on(corpus: str) -> Run:
    return Run(
        configuration=configuration(
            name=f"{corpus}-test",
            budget=budget(epochs=2, batch_size=32, warmup_epochs=1, final_lr_fraction=0.01),
        ),
        corpus=corpus,
        units=16,
        device="cpu",
    )


def truth_spectrum(corpus: str, workspace: Path) -> Spectrum:
    """The energy of the channels hidden whole in a corpus's validation windows, per frequency."""
    run = run_on(corpus)
    published = publish(run, workspace)
    fitted = SpectralRecovery.up_to(published.spectral_cycles())
    for batch, masks, _ in validation_batches(published, run):
        fitted = fitted.observe(batch, masks, torch.zeros_like(batch.features[..., 0]))
    return Spectrum(
        fitted.truth_energy, fitted.residual_energy, fitted.residual_energy, fitted.fitted, 0
    )


def test_the_probe_is_the_dense_control_with_only_the_band_turned() -> None:
    assert SPECTRAL_PROBE.model_dump() == {
        **CONTROL_A.model_dump(),
        "name": "spectral-probe",
        "units": 80,
    }
    assert SPECTRAL_PROCESS.model_dump() == {
        **CONTROL_PROCESS.model_dump(),
        "shortest_period": 6.0,
        "longest_period": 32.0,
    }


def test_the_probe_stays_out_of_the_controls_registry_but_the_report_trains_on_it() -> None:
    assert SPECTRAL_PROBE.name not in LAYOUTS
    assert SPECTRAL_PROBE.name not in KnownCorpora.default().names()
    assert SPECTRAL_PROBE.name in corpora()
    assert set(LAYOUTS) <= set(corpora())
    assert SPECTRAL_PROBE.name in {file.corpus for file in experiments().values()}


def test_the_report_generates_the_probe_from_its_own_process_under_the_generated_terms() -> None:
    process, layout, known = generated(SPECTRAL_PROBE.name)
    control_process, control_layout, control_known = generated("control-a")

    assert (process, layout) == (SPECTRAL_PROCESS, SPECTRAL_PROBE)
    assert (control_process, control_layout) == (CONTROL_PROCESS, CONTROL_A)
    assert (known.source, known.licence) == (control_known.source, control_known.licence)


def test_every_harmonic_lies_inside_what_the_report_fits_and_samples_well() -> None:
    cycles = WINDOW.length * SPECTRAL_PROCESS.frequencies()
    periods_in_samples = 1.0 / (SPECTRAL_PROCESS.frequencies() * SPECTRAL_PROBE.time_step)

    assert SPECTRAL_PROBE.cadence == 1
    assert cycles.min() >= 1.0
    assert cycles.max() <= CYCLES_AT_MOST
    assert periods_in_samples.min() >= 5.0
    assert len(np.unique(np.round(cycles))) >= 3


def test_the_probe_spreads_a_hidden_channel_over_several_frequencies_where_the_control_cannot(
    tmp_path: Path,
) -> None:
    probe = truth_spectrum(SPECTRAL_PROBE.name, tmp_path / "probe")
    control = truth_spectrum("control-a", tmp_path / "control")

    assert len(control.informative()) == 1
    assert len(probe.informative()) >= 3
    assert probe.shares()[0] < 0.8
