import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.grid_reading import GridReading  # noqa: E402
from tests.evaluation.adapters.features.support import timed, window  # noqa: E402

pytestmark = pytest.mark.ml

EVERY_STEP = [(1.0, 0.1), (2.0, 0.3), (3.0, 0.6), (4.0, 0.9)]


def test_only_the_channels_the_windows_observe_are_read() -> None:
    fitted = [window(timed(1, EVERY_STEP), timed(3, [(5.0, 0.4)]))]

    assert GridReading.over(fitted, steps=4, channels=4).held == (0, 2)


def test_every_channel_read_brings_its_mask_even_one_never_missed_before() -> None:
    reading = GridReading.over([window(timed(1, EVERY_STEP))], steps=4, channels=2)
    later = window(timed(1, [(7.0, 0.1)]))

    values, observed = reading.tensors([later])

    assert values.shape == observed.shape == (1, 1, 4)
    assert observed[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert values[0, 0].tolist() == [7.0, 7.0, 7.0, 7.0]
