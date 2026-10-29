import pytest

from emblema.evaluation.domain.exceptions import InvalidOutcomeError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.target_kind import TargetKind

SCHEME = AnomalyScheme()


def test_the_target_is_the_events_number_and_zero_for_none() -> None:
    assert SCHEME.target(event=0.0) == 0.0
    assert SCHEME.target(event=3.0) == 3.0
    assert SCHEME.kind is TargetKind.ANOMALY
    assert SCHEME.scale == 1.0


def test_a_window_the_ground_truth_excludes_is_not_scored_and_has_no_target() -> None:
    assert not SCHEME.scores(AnomalyScheme.EXCLUDED)
    assert SCHEME.scores(0.0)
    assert SCHEME.scores(2.0)
    with pytest.raises(InvalidOutcomeError, match="numbered from one"):
        SCHEME.target(event=AnomalyScheme.EXCLUDED)


@pytest.mark.parametrize("event", [0.5, -2.0, float("nan")])
def test_anything_but_a_whole_non_negative_number_is_no_event(event: float) -> None:
    with pytest.raises(InvalidOutcomeError, match="numbered from one"):
        SCHEME.target(event=event)
