import math
from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.diagnostics.spectral_recovery import (  # noqa: E402
    SpectralRecovery,
)
from tests.pretraining.adapters.diagnostics.conftest import batch_of, hiding, window  # noqa: E402

pytestmark = pytest.mark.ml

CYCLES = 4


def tone(cycles: int, amplitude: float = 1.0) -> Callable[[float], float]:
    return lambda t: amplitude * math.sin(2 * math.pi * cycles * t)


def test_a_perfect_prediction_recovers_every_frequency_the_truth_holds() -> None:
    batch = batch_of(window({1: tone(3), 2: tone(1)}, steps=40))
    masks = hiding(batch, channels=[1])
    truth = batch.features[..., 0]

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, truth)

    assert recovery.fitted == 1
    assert recovery.recovered()[2] == pytest.approx(1.0)
    # The truth holds energy at three cycles only, so the other shares are undefined and zero.
    assert [energy > 1e-6 for energy in recovery.truth_energy] == [False, False, True, False]
    assert recovery.truth_energy[2] == pytest.approx(1.0, abs=1e-6)


def test_a_prediction_of_nothing_recovers_nothing() -> None:
    batch = batch_of(window({1: tone(3)}, steps=40))
    masks = hiding(batch, channels=[1])
    nothing = torch.zeros(batch.padding_mask.shape)

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, nothing)

    assert recovery.recovered()[2] == pytest.approx(0.0)


def test_a_smooth_prediction_recovers_the_slow_component_and_loses_the_fast_one() -> None:
    slow, fast = tone(1, 2.0), tone(4, 1.0)
    batch = batch_of(window({1: lambda t: slow(t) + fast(t)}, steps=48))
    masks = hiding(batch, channels=[1])
    smooth = torch.tensor([[slow(t) for t in batch.timestamps[0].tolist()]])

    recovered = SpectralRecovery.up_to(CYCLES).observe(batch, masks, smooth).recovered()

    assert recovered[0] == pytest.approx(1.0, abs=1e-6)
    assert recovered[3] == pytest.approx(0.0, abs=1e-6)


def test_a_trend_and_an_offset_are_not_frequencies() -> None:
    batch = batch_of(window({1: lambda t: 3.0 + 2.0 * t + tone(2)(t)}, steps=40))
    masks = hiding(batch, channels=[1])
    nothing = torch.zeros(batch.padding_mask.shape)

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, nothing)

    assert recovery.truth_energy[1] == pytest.approx(1.0, abs=1e-6)
    assert sum(recovery.truth_energy) == pytest.approx(1.0, abs=1e-6)


def test_the_fit_is_well_conditioned_on_an_irregular_sampling() -> None:
    generator = torch.Generator().manual_seed(3)
    instants = sorted(torch.rand(30, generator=generator).tolist())
    batch = batch_of(window({1: tone(2)}, steps=30, instants=instants))
    masks = hiding(batch, channels=[1])

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, batch.features[..., 0])

    assert recovery.fitted == 1
    assert recovery.truth_energy[1] == pytest.approx(1.0, rel=0.05)
    assert sum(recovery.truth_energy) == pytest.approx(1.0, rel=0.1)


def test_a_prediction_that_doubles_the_signal_recovers_nothing_of_it() -> None:
    batch = batch_of(window({1: tone(2)}, steps=40))
    masks = hiding(batch, channels=[1])
    louder = 2.0 * batch.features[..., 0]

    recovered = SpectralRecovery.up_to(CYCLES).observe(batch, masks, louder).recovered()

    # The residual is the negative of the truth: the same energy, so nothing is recovered.
    assert recovered[1] == pytest.approx(0.0)


def test_only_channels_hidden_whole_are_fitted_and_short_ones_are_skipped() -> None:
    wide = window({1: tone(1), 2: tone(2)}, steps=40)
    narrow = window({1: tone(1)}, steps=6)
    batch = batch_of(wide, narrow)
    where = (batch.channel_ids == 2) & (batch.timestamps < 0.5)
    masks = hiding(batch, channels=[1], where=where)
    truth = batch.features[..., 0]

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, truth)

    assert (recovery.fitted, recovery.skipped) == (1, 1)


def test_timeless_tokens_have_no_spectrum() -> None:
    batch = batch_of(window({1: tone(1)}, steps=40, timeless={9: 2.0}))
    masks = hiding(batch, channels=[9])

    recovery = SpectralRecovery.up_to(CYCLES).observe(batch, masks, batch.features[..., 0])

    assert (recovery.fitted, recovery.skipped) == (0, 0)


def test_the_tallies_add_across_batches() -> None:
    first = batch_of(window({1: tone(2)}, steps=40))
    second = batch_of(window({1: tone(2, 3.0)}, steps=40))
    zero = torch.zeros(first.padding_mask.shape)

    recovery = (
        SpectralRecovery.up_to(CYCLES)
        .observe(first, hiding(first, channels=[1]), zero)
        .observe(second, hiding(second, channels=[1]), zero)
    )

    assert recovery.fitted == 2
    assert recovery.truth_energy[1] == pytest.approx(1.0 + 9.0)
    assert recovery.residual_energy == recovery.truth_energy


def test_how_many_cycles_a_channel_of_so_many_tokens_can_carry() -> None:
    assert SpectralRecovery.cycles_fitting(32) == 7
    assert SpectralRecovery.cycles_fitting(10) == 1
    assert SpectralRecovery.cycles_fitting(5) == 0
    assert SpectralRecovery.up_to(7).coefficients == 16


def test_the_frequency_count_and_the_energies_are_validated() -> None:
    with pytest.raises(ValueError, match="cycles"):
        SpectralRecovery.up_to(0)
    with pytest.raises(ValueError, match="one entry per frequency"):
        SpectralRecovery(2, (0.0,), (0.0, 0.0), fitted=0, skipped=0)
