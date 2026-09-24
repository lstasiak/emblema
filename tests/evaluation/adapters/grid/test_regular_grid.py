import numpy as np
import pytest

from emblema.evaluation.adapters.grid.regular_grid import RegularGrid
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.kernel.tokens import Token
from tests.evaluation.adapters.features.support import timed, window


def test_a_regular_window_on_as_many_steps_as_readings_is_laid_without_loss() -> None:
    readings = [(float(k), (k + 0.5) / 8) for k in range(8)]

    laid = RegularGrid(8, 1).of([window(timed(1, readings))])[0]

    assert laid[0].tolist() == [float(k) for k in range(8)]
    assert laid[1].tolist() == [1.0] * 8


@pytest.mark.parametrize("readings", [50, 64, 128])
def test_readings_stored_at_the_start_of_every_step_in_single_precision_fill_every_step(
    readings: int,
) -> None:
    # Times as a published window stores them: k / n in float32, a hair below k / n for some k.
    times = (np.arange(readings, dtype=np.float32) / np.float32(readings)).tolist()

    laid = RegularGrid(readings, 1).of(
        [window(timed(1, [(float(k), t) for k, t in enumerate(times)]))]
    )[0]

    assert laid[1].tolist() == [1.0] * readings
    assert laid[0].tolist() == [float(k) for k in range(readings)]


def test_a_step_without_a_reading_carries_the_last_one_and_says_it_was_not_observed() -> None:
    laid = RegularGrid(4, 1).of([window(timed(1, [(3.0, 0.1), (7.0, 0.8)]))])[0]

    assert laid[0].tolist() == [3.0, 3.0, 3.0, 7.0]
    assert laid[1].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_a_channel_not_yet_observed_holds_the_mean_until_it_is() -> None:
    laid = RegularGrid(4, 2).of([window(timed(1, [(1.0, 0.1)]), timed(2, [(5.0, 0.6)]))])[0]

    assert laid[1].tolist() == [0.0, 0.0, 5.0, 5.0]
    assert laid[3].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_the_latest_reading_of_a_step_is_the_one_that_stands() -> None:
    laid = RegularGrid(2, 1).of([window(timed(1, [(1.0, 0.1), (2.0, 0.2), (9.0, 0.3)]))])[0]

    assert laid[0, 0] == 9.0


def test_a_reading_at_the_very_end_falls_in_the_last_step() -> None:
    laid = RegularGrid(4, 1).of([window(timed(1, [(6.0, 1.0)]))])[0]

    assert laid[0].tolist() == [0.0, 0.0, 0.0, 6.0]


def test_a_static_feature_holds_its_value_at_every_step() -> None:
    static = [Token(2, 1.5, 0.0, 0.0, timeless=True)]

    laid = RegularGrid(3, 2).of([window(static, timed(1, [(1.0, 0.5)]))])[0]

    assert laid[1].tolist() == [1.5, 1.5, 1.5]
    assert laid[3].tolist() == [1.0, 1.0, 1.0]


def test_windows_of_one_corpus_stack_whatever_they_hold() -> None:
    laid = RegularGrid(5, 3).of([window(timed(1, [(1.0, 0.2)])), window(timed(3, [(2.0, 0.9)]))])

    assert laid.shape == (2, 6, 5)
    assert np.isfinite(laid).all()


def test_a_channel_beyond_the_vocabulary_is_refused() -> None:
    with pytest.raises(UnreadableTaskCorpusError, match="holds channel 3"):
        RegularGrid(4, 2).of([window(timed(3, [(1.0, 0.5)]))])


@pytest.mark.parametrize(("steps", "channels"), [(0, 1), (4, 0)])
def test_a_grid_of_no_step_or_no_channel_is_refused(steps: int, channels: int) -> None:
    with pytest.raises(ValueError, match="steps and channels"):
        RegularGrid(steps, channels)
