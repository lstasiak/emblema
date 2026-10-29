import pytest

from emblema.evaluation.domain.detection.detection_windows import DetectionWindows
from emblema.evaluation.domain.exceptions import InvalidScoredOutcomeError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from tests.evaluation.support import labelled, task, window

DETECTION = task(protocol=EvaluationProtocol.ANOMALY_DETECTION, labels=AnomalyScheme(), strata=None)


def test_a_run_needs_a_window_on_each_side() -> None:
    with pytest.raises(InvalidScoredOutcomeError, match="fit on"):
        DetectionWindows(task=DETECTION, fitting=(), scored=(labelled("c", 0, 1.0, 0.0),))
    with pytest.raises(InvalidScoredOutcomeError, match="answer"):
        DetectionWindows(task=DETECTION, fitting=(window("a", 0, 1.0),), scored=())
