from math import sqrt

import pytest

from emblema.evaluation.domain.exceptions import InvalidDetectionThresholdError
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold


def test_a_gain_carries_the_spread_of_its_pretrained_side_alone() -> None:
    threshold = DetectionThreshold.of_gain(standard_error=0.005, spread=0.014, pretrainings=2)

    assert threshold.value == pytest.approx(2 * sqrt(0.005**2 + (0.014**2 / 2) / 2))


def test_more_pretrainings_lower_the_threshold_toward_the_seeds_alone() -> None:
    def at(pretrainings: int) -> float:
        return DetectionThreshold.of_gain(
            standard_error=0.005, spread=0.014, pretrainings=pretrainings
        ).value

    assert at(1) > at(2) > at(8) > 2 * 0.005


def test_a_contrast_carries_the_spread_of_both_sides() -> None:
    threshold = DetectionThreshold.of_contrast(standard_error=0.005, spread=0.02, sides=(4, 4))

    assert threshold.value == pytest.approx(2 * sqrt(0.005**2 + (0.02**2 / 2) * 0.5))


@pytest.mark.parametrize(
    ("standard_error", "spread", "sides"),
    [(-0.001, 0.014, (2,)), (0.005, float("nan"), (2,)), (0.005, 0.014, (0,)), (0.005, 0.014, ())],
)
def test_a_threshold_out_of_range_is_refused(
    standard_error: float, spread: float, sides: tuple[int, ...]
) -> None:
    with pytest.raises(InvalidDetectionThresholdError):
        DetectionThreshold.of_contrast(standard_error=standard_error, spread=spread, sides=sides)


def test_a_negative_value_is_refused() -> None:
    with pytest.raises(InvalidDetectionThresholdError, match="not negative"):
        DetectionThreshold(value=-0.01)
