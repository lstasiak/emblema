import pytest

from emblema.evaluation.domain.detection.detection_windows import DetectionWindows
from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError, InvalidTaughtEventsError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from tests.evaluation.support import labelled, task, window

DETECTION = task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None)


def test_a_run_needs_a_window_on_each_side() -> None:
    with pytest.raises(InvalidScoredOutcomeError, match="fit on"):
        DetectionWindows(task=DETECTION, fitting=(), scored=(labelled("c", 0, 1.0, 0.0),))
    with pytest.raises(InvalidScoredOutcomeError, match="answer"):
        DetectionWindows(task=DETECTION, fitting=(window("a", 0, 1.0),), scored=())


def test_a_taught_window_is_a_fitting_window_that_overlaps_an_event() -> None:
    fitting = (window("a", 0, 1.0), window("a", 1, 2.0))
    scored = (labelled("c", 5, 1.0, 0.0),)

    windows = DetectionWindows(
        task=DETECTION, fitting=fitting, scored=scored, taught=(labelled("a", 1, 2.0, 3.0),)
    )

    assert windows.taught[0].target == 3.0
    with pytest.raises(InvalidTaughtEventsError, match="not on the fitting side"):
        DetectionWindows(
            task=DETECTION, fitting=fitting, scored=scored, taught=(labelled("c", 5, 1.0, 1.0),)
        )
    with pytest.raises(InvalidTaughtEventsError, match="overlaps an event"):
        DetectionWindows(
            task=DETECTION, fitting=fitting, scored=scored, taught=(labelled("a", 0, 1.0, 0.0),)
        )
