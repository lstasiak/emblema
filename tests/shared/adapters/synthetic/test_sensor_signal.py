import numpy as np
import pytest

from emblema.shared.adapters.synthetic.layouts import CONTROL_B
from emblema.shared.adapters.synthetic.sensor_signal import PRECISION, SensorSignal
from tests.support.synthetic import HOSTILE, PROCESS, miniature

LAYOUT = miniature(CONTROL_B)


def test_the_signal_is_what_a_noiseless_sensor_reports() -> None:
    from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
    from emblema.catalog.domain.identifiers import UnitKey

    layout = LAYOUT.with_dials(noise=0.0, missing=0.0)
    signal = SensorSignal(PROCESS, layout)
    reader = SyntheticCorpusReader(PROCESS, layout)

    reported = [
        o
        for o in reader.read_observations(UnitKey.within(layout.name, "0"))
        if o.channel == layout.channel_names[1]
    ]
    times = np.array([o.time for o in reported])
    exact = signal.values_at(0, 1, times)

    assert reported, "the first unit reports on its second channel"
    assert np.round(exact, PRECISION).tolist() == pytest.approx([o.value for o in reported])


def test_the_signal_answers_at_instants_no_sensor_reported() -> None:
    signal = SensorSignal(PROCESS, LAYOUT)

    between = signal.values_at(0, 0, np.array([0.5, 1.5, 1000.25]))

    assert between.shape == (3,)
    assert np.isfinite(between).all()


def test_the_gain_is_the_unit_gain_at_the_corpus_precision() -> None:
    signal = SensorSignal(PROCESS, LAYOUT)

    gain = signal.gain(0)

    assert round(gain, PRECISION) == gain
    assert 1.0 - LAYOUT.gain_spread <= gain <= 1.0 + LAYOUT.gain_spread


def test_turning_the_coupling_off_keeps_the_signal_strength() -> None:
    coupled = SensorSignal(PROCESS, LAYOUT)
    uncoupled = SensorSignal(PROCESS, LAYOUT.with_dials(name="null", coupling=0.0))
    times = np.arange(0.0, 200.0)

    on, off = coupled.values_at(0, 0, times), uncoupled.values_at(0, 0, times)

    assert not np.allclose(on, off)
    assert np.std(off) == pytest.approx(np.std(on), rel=0.5)


def test_a_channel_cannot_respond_to_more_factors_than_there_are() -> None:
    with pytest.raises(ValueError, match="cannot respond"):
        SensorSignal(PROCESS, HOSTILE.with_dials(factors_per_channel=PROCESS.factors + 1))
