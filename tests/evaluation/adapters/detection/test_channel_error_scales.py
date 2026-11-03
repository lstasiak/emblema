import numpy as np
import pytest

from emblema.evaluation.adapters.detection.channel_error_scales import ChannelErrorScales
from emblema.evaluation.adapters.detection.channel_errors import ChannelErrors

CHANNELS = np.array([1, 2, 3], dtype=np.int64)


def errors(missed: list[list[float]], counted: list[list[int]]) -> ChannelErrors:
    return ChannelErrors(
        channels=CHANNELS, missed=np.array(missed), counted=np.array(counted, dtype=np.int64)
    )


# Four fitting windows: the first channel missed by one, the second by four, the third never
# scored.
FITTING = errors(
    [[1.0, 4.0, 0.0], [1.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
    [[1, 1, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]],
)


def test_a_channel_s_scale_is_its_median_error_over_the_windows_that_scored_it() -> None:
    scales = ChannelErrorScales.fitted_on(FITTING)

    assert scales.scales[:2].tolist() == [1.0, 4.0]


def test_a_channel_missed_by_nothing_takes_the_floor_and_one_never_scored_the_median() -> None:
    scales = ChannelErrorScales.fitted_on(FITTING)
    exact = ChannelErrorScales.fitted_on(
        errors([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]], [[1, 1, 0], [1, 1, 0]])
    )

    # The floor is a tenth of the median over the channels that were scored; the channel never
    # scored is unknown, not quiet, and takes that median.
    assert scales.scales[2] == pytest.approx(2.5)
    assert exact.scales.tolist() == pytest.approx([5.0, 100.0, 50.0])


def test_where_the_scales_median_is_nought_the_floor_is_the_smallest_scale_above_it() -> None:
    mostly_exact = ChannelErrorScales.fitted_on(
        errors([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]], [[1, 1, 1], [1, 1, 1]])
    )
    all_exact = ChannelErrorScales.fitted_on(errors([[0.0, 0.0, 0.0]], [[1, 1, 1]]))

    assert mostly_exact.scales.tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert all_exact.scales.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_a_window_s_error_is_the_mean_over_its_channels_of_each_over_its_scale() -> None:
    scales = ChannelErrorScales.fitted_on(FITTING)
    scored = errors(
        [[4.0, 100.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
    )

    assert scales.read(scored).tolist() == pytest.approx([(4.0 + 25.0) / 2, 2.0, 0.0])
    # A window scoring the channel never scored on the fitting side reads it over the median.
    only_unknown = errors([[0.0, 0.0, 5.0]], [[0, 0, 1]])
    assert scales.read(only_unknown).tolist() == pytest.approx([2.0])


def test_errors_on_other_channels_are_refused() -> None:
    scales = ChannelErrorScales.fitted_on(FITTING)
    other = ChannelErrors(
        channels=np.array([1, 2], dtype=np.int64),
        missed=np.ones((1, 2)),
        counted=np.ones((1, 2), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="other channels"):
        scales.read(other)


def test_a_scale_is_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        ChannelErrorScales(channels=CHANNELS, scales=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="one scale per channel"):
        ChannelErrorScales(channels=CHANNELS, scales=np.array([1.0, 1.0]))
