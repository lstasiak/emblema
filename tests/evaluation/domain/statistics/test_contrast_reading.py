from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.contrast_reading import ContrastReading
from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.paired_reading import PairedReading


def contrast(mean: float, *, decides: bool) -> ContrastReading:
    return ContrastReading(
        reading=PairedReading(
            mean=mean,
            standard_error=0.005,
            interval=BootstrapInterval(low=mean - 0.01, high=mean + 0.01, level=0.95),
        ),
        threshold=DetectionThreshold(value=0.02),
        decides=decides,
    )


def test_a_contrast_above_its_threshold_where_a_budget_decides_moves_the_recipe() -> None:
    assert ContrastReading.moves([contrast(0.0, decides=False), contrast(0.05, decides=True)])


def test_a_contrast_above_only_where_no_budget_decides_leaves_the_reference() -> None:
    assert not ContrastReading.moves([contrast(0.05, decides=False), contrast(0.0, decides=True)])


def test_a_harm_anywhere_leaves_the_reference() -> None:
    assert not ContrastReading.moves([contrast(-0.05, decides=False), contrast(0.05, decides=True)])
    assert contrast(-0.05, decides=False).below
