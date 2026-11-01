import numpy as np
import pytest

from emblema.evaluation.adapters.detection.channel_errors import ChannelErrors

CHANNELS = np.array([1, 2], dtype=np.int64)
# Three windows: the first scored on both channels, the second on the first alone, the third
# on none.
ERRORS = ChannelErrors(
    channels=CHANNELS,
    missed=np.array([[2.0, 30.0], [3.0, 0.0], [0.0, 0.0]]),
    counted=np.array([[2, 3], [1, 0], [0, 0]], dtype=np.int64),
)


def test_the_raw_error_is_one_mean_over_every_scored_cell() -> None:
    assert ERRORS.raw().tolist() == pytest.approx([32.0 / 5, 3.0, 0.0])


def test_the_error_per_channel_is_a_mean_per_channel_and_nothing_where_none_was_scored() -> None:
    per_channel = ERRORS.per_channel()

    assert per_channel[0].tolist() == pytest.approx([1.0, 10.0])
    assert per_channel[1, 0] == 3.0
    assert np.isnan(per_channel[1, 1])
    assert np.isnan(per_channel[2]).all()
    assert ERRORS.unread.tolist() == [False, False, True]


def test_sums_counts_and_channels_lie_on_one_grid() -> None:
    with pytest.raises(ValueError, match="one grid"):
        ChannelErrors(
            channels=CHANNELS, missed=np.zeros((2, 3)), counted=np.zeros((2, 3), dtype=np.int64)
        )
