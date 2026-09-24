import numpy as np
import pytest

from emblema.evaluation.adapters.synthetic.synthetic_ground_truth import SyntheticGroundTruth
from emblema.evaluation.domain.exceptions import UnknownGroundTruthError
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.shared.adapters.synthetic.layouts import CONTROL_B
from emblema.shared.adapters.synthetic.sensor_signal import PRECISION, SensorSignal
from tests.evaluation.support import window
from tests.support.synthetic import PROCESS, miniature

LAYOUT = miniature(CONTROL_B)
SIGNAL = SensorSignal(PROCESS, LAYOUT)


def test_the_truth_is_the_exact_reading_of_the_named_sensor_at_the_horizon() -> None:
    truth = SyntheticGroundTruth(SIGNAL, ForecastScheme("s02", 12.0))
    asked = window(f"{LAYOUT.name}/1", 0, 40.0)

    expected = round(float(SIGNAL.values_at(1, 1, np.array([52.0]))[0]), PRECISION)
    assert truth.truths_of(LAYOUT.name, [asked]) == {asked: expected}


def test_a_horizon_of_zero_reads_the_sensor_at_the_windows_end() -> None:
    now = SyntheticGroundTruth(SIGNAL, ForecastScheme("s01", 0.0))
    later = SyntheticGroundTruth(SIGNAL, ForecastScheme("s01", 12.0))
    at_forty, at_fifty_two = (
        window(f"{LAYOUT.name}/0", 0, 40.0),
        window(f"{LAYOUT.name}/0", 1, 52.0),
    )

    assert (
        now.truths_of(LAYOUT.name, [at_fifty_two])[at_fifty_two]
        == later.truths_of(LAYOUT.name, [at_forty])[at_forty]
    )


def test_a_sensor_the_layout_does_not_have_is_refused_before_any_window_is_asked() -> None:
    with pytest.raises(UnknownGroundTruthError, match="no channel"):
        SyntheticGroundTruth(SIGNAL, ForecastScheme("s99", 12.0))


@pytest.mark.parametrize("name", ["control-a/0", f"{LAYOUT.name}/00", f"{LAYOUT.name}/x", "0"])
def test_a_key_of_another_layout_or_of_another_shape_names_no_unit(name: str) -> None:
    truth = SyntheticGroundTruth(SIGNAL, ForecastScheme("s01", 12.0))

    with pytest.raises(UnknownGroundTruthError, match="not a unit"):
        truth.truths_of(LAYOUT.name, [window(name, 0, 40.0)])


def test_the_coupled_and_the_null_layout_answer_differently_about_the_same_unit() -> None:
    coupled = SyntheticGroundTruth(SIGNAL, ForecastScheme("s01", 12.0))
    null_layout = LAYOUT.with_dials(name="null-miniature", coupling=0.0)
    uncoupled = SyntheticGroundTruth(
        SensorSignal(PROCESS, null_layout), ForecastScheme("s01", 12.0)
    )

    asked = window(f"{LAYOUT.name}/0", 0, 40.0)
    asked_null = window(f"{null_layout.name}/0", 0, 40.0)

    assert (
        coupled.truths_of(LAYOUT.name, [asked])[asked]
        != uncoupled.truths_of(null_layout.name, [asked_null])[asked_null]
    )
