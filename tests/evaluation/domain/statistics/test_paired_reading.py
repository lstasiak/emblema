import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedReadingError
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.outcome_level import OutcomeLevel
from emblema.evaluation.domain.statistics.paired_reading import PairedReading

THRESHOLD = DetectionThreshold(value=0.015)


def reading(mean: float, half: float = 0.01, standard_error: float = 0.005) -> PairedReading:
    return PairedReading(
        mean=mean,
        standard_error=standard_error,
        interval=BootstrapInterval(low=mean - half, high=mean + half, level=0.95),
    )


def test_its_noise_is_the_seeds_variance_and_the_units_from_the_interval() -> None:
    assert reading(0.0, half=0.0196, standard_error=0.004).noise == pytest.approx(
        0.004**2 + 0.01**2
    )


@pytest.mark.parametrize(
    ("mean", "half", "level"),
    [
        (0.025, 0.01, OutcomeLevel.LEAST_GAIN),
        (0.02, 0.01, OutcomeLevel.LEAST_GAIN),
        (0.018, 0.01, OutcomeLevel.REAL_AND_SMALL),
        (0.014, 0.005, OutcomeLevel.NOISE),
        (0.03, 0.04, OutcomeLevel.NOISE),
        (-0.03, 0.01, OutcomeLevel.HARM),
        (-0.03, 0.04, OutcomeLevel.NOISE),
        (-0.014, 0.005, OutcomeLevel.NOISE),
    ],
)
def test_a_gain_reads_its_level_from_its_interval_threshold_and_the_least_gain(
    mean: float, half: float, level: OutcomeLevel
) -> None:
    assert reading(mean, half=half).level(THRESHOLD, least_gain=0.02) is level


def test_a_mean_exactly_at_the_threshold_is_not_detected() -> None:
    at = reading(0.015, half=0.005)

    assert not at.exceeds(THRESHOLD)
    assert not reading(-0.015).falls_below(THRESHOLD)


@pytest.mark.parametrize(("mean", "standard_error"), [(float("inf"), 0.0), (0.0, -0.001)])
def test_a_reading_out_of_range_is_refused(mean: float, standard_error: float) -> None:
    with pytest.raises(InvalidPairedReadingError):
        PairedReading(
            mean=mean,
            standard_error=standard_error,
            interval=BootstrapInterval(low=-0.01, high=0.01, level=0.95),
        )
