import numpy as np
import pytest

from emblema.evaluation.adapters.detection.channel_local_detector import ChannelLocalDetector
from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError

DEPARTURES = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [0.0, 4.0, 1.0]])
PRESENT = np.array(
    [[True, True, False], [False, False, False], [True, True, True], [True, True, True]]
)
FLOOR = np.full((3, 5), 0.5)


def test_a_windows_score_is_the_largest_departure_or_the_mean_over_the_present() -> None:
    assert ChannelLocalDetector.reduced(DEPARTURES, PRESENT, ChannelReduction.MAX).tolist() == [
        3.0,
        0.0,
        2.0,
        4.0,
    ]
    assert ChannelLocalDetector.reduced(
        DEPARTURES, PRESENT, ChannelReduction.MEAN
    ).tolist() == pytest.approx([2.0, 0.0, 2.0, 5.0 / 3.0])


@pytest.mark.parametrize("channels", list(ChannelReduction))
def test_the_alarm_is_set_over_the_fitting_windows_own_scores(channels: ChannelReduction) -> None:
    detector = ChannelLocalDetector.fitted_on(
        DEPARTURES,
        PRESENT,
        history=ChannelHistory(channels=channels),
        floor=FLOOR,
        contamination=0.25,
    )

    expected = ChannelLocalDetector.reduced(DEPARTURES, PRESENT, channels)
    assert detector.alarm.level == pytest.approx(np.quantile(expected, 0.75))
    np.testing.assert_allclose(
        detector.scores(DEPARTURES, PRESENT), expected - detector.alarm.level
    )


def test_a_detector_read_against_units_needs_the_units_of_what_it_scores() -> None:
    units = ["a", "a", "b", "b"]
    detector = ChannelLocalDetector.fitted_on(
        DEPARTURES,
        PRESENT,
        history=ChannelHistory(),
        floor=FLOOR,
        contamination=0.25,
        units=units,
    )

    assert detector.scores(DEPARTURES, PRESENT, units=units).shape == (4,)
    with pytest.raises(ValueError, match="given no units"):
        detector.scores(DEPARTURES, PRESENT)


def test_the_kept_detector_reads_back_as_itself() -> None:
    detector = ChannelLocalDetector.fitted_on(
        DEPARTURES,
        PRESENT,
        history=ChannelHistory(
            span=24, departure=DepartureCells.BINS, channels=ChannelReduction.MEAN
        ),
        floor=FLOOR,
        contamination=0.25,
        units=["a", "a", "b", "b"],
    )

    kept = detector.kept()
    back = ChannelLocalDetector.from_bytes(kept.content)

    assert kept.format == ChannelLocalDetector.FORMAT
    np.testing.assert_array_equal(back.floor, FLOOR)
    assert (back.history, back.alarm) == (detector.history, detector.alarm)


def test_bytes_that_are_not_its_archive_are_refused() -> None:
    with pytest.raises(UnreadableFittedCandidateError, match="channel-local"):
        ChannelLocalDetector.from_bytes(b"not an archive")
