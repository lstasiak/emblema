import math

import numpy as np
import pytest
from scipy.signal import lombscargle

from emblema.evaluation.adapters.features.window_spectrum import WindowSpectrum
from emblema.shared.kernel.tokens import Token
from tests.evaluation.adapters.features.support import timed, window


def sine(channel: int, cycles: float, times: np.ndarray, offset: float = 0.0) -> list[Token]:
    return timed(
        channel, [(float(np.sin(2 * np.pi * cycles * t) + offset), float(t)) for t in times]
    )


def at(name: str) -> int:
    return WindowSpectrum.NAMES.index(name)


REGULAR = np.arange(1, 129) / 128
IRREGULAR = np.sort(np.random.default_rng(3).uniform(0.01, 1.0, 60))


def test_the_power_is_the_floating_mean_periodogram_of_the_readings_as_they_were_taken() -> None:
    readings = sine(4, 5.0, IRREGULAR, offset=2.0)
    times = np.array([token.time for token in readings])
    values = np.array([token.value for token in readings])
    cycles = WindowSpectrum.CYCLES
    resolved = cycles[len(readings) / 2 >= cycles]
    reference = lombscargle(times, values, 2 * np.pi * resolved, normalize=True, floating_mean=True)

    row = WindowSpectrum.of(window(readings)).rows[0]

    assert row[at("peak")] == pytest.approx(reference.max())
    share = reference / reference.sum()
    assert row[at("centroid")] == pytest.approx(share @ resolved)


@pytest.mark.parametrize("times", [REGULAR, IRREGULAR], ids=["regular", "irregular"])
def test_a_pure_oscillation_puts_its_power_in_its_own_octave(times: np.ndarray) -> None:
    row = WindowSpectrum.of(window(sine(1, 5.0, times))).rows[0]

    assert row[at("band_4_8")] > 0.5
    assert row[at("peak")] > 0.9


def test_the_octaves_share_all_of_the_power_between_them() -> None:
    row = WindowSpectrum.of(window(sine(1, 3.0, IRREGULAR, offset=1.0))).rows[0]

    assert sum(row[at(name)] for name in WindowSpectrum.NAMES[:5]) == pytest.approx(1.0)
    assert 0.0 <= row[at("entropy")] <= 1.0


def test_a_slow_channel_is_centred_lower_than_a_fast_one() -> None:
    spectrum = WindowSpectrum.of(window(sine(1, 2.0, REGULAR), sine(2, 20.0, REGULAR)))

    slow, fast = spectrum.rows[:, at("centroid")]
    assert slow < fast
    assert spectrum.channels.tolist() == [1, 2]


def test_a_channel_holds_no_power_above_the_frequency_its_readings_resolve() -> None:
    # Twenty evenly spaced readings resolve ten cycles per window; a pure tone at five is then
    # read in its own octave with nothing aliased above the limit.
    row = WindowSpectrum.of(window(sine(1, 5.0, np.arange(1, 21) / 20))).rows[0]

    assert row[at("band_16_32")] == 0.0
    assert row[at("band_4_8")] > 0.9
    assert row[at("centroid")] <= 10.0


def test_a_channel_resolving_a_single_frequency_has_no_entropy() -> None:
    row = WindowSpectrum.of(window(timed(1, [(1.0, 0.2), (3.0, 0.5), (2.0, 0.9)]))).rows[0]

    assert row[at("band_1_2")] == pytest.approx(1.0)
    assert math.isnan(row[at("entropy")])


@pytest.mark.parametrize(
    "readings",
    [
        timed(1, [(1.0, 0.2), (2.0, 0.5)]),
        timed(1, [(3.0, 0.2), (3.0, 0.5), (3.0, 0.9)]),
    ],
    ids=["two readings", "no variation"],
)
def test_a_channel_that_cannot_support_a_spectrum_has_none(readings: list[Token]) -> None:
    row = WindowSpectrum.of(window(readings)).rows[0]

    assert all(math.isnan(value) for value in row)


def test_a_static_feature_has_no_spectrum_and_no_row() -> None:
    static = [Token(7, 1.5, 0.0, 0.0, timeless=True)]

    spectrum = WindowSpectrum.of(window(static, sine(2, 4.0, REGULAR)))

    assert spectrum.channels.tolist() == [2]
