import pytest

from emblema.evaluation.domain.detection.channel_history import ChannelHistory
from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction
from emblema.evaluation.domain.detection.departure_cells import DepartureCells
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError


def test_the_standard_history_is_six_windows_read_by_the_summary_and_the_largest_channel() -> None:
    history = ChannelHistory()

    assert (history.span, history.departure, history.channels) == (
        6,
        DepartureCells.WINDOW,
        ChannelReduction.MAX,
    )
    assert history.windows_before == 5


@pytest.mark.parametrize(("span", "least"), [(2, 1), (4, 2), (6, 3), (24, 12), (72, 36)])
def test_a_channel_needs_half_the_earlier_windows_rounded_up(span: int, least: int) -> None:
    assert ChannelHistory(span=span).least_history == least


@pytest.mark.parametrize("span", [1, 0, -6])
def test_a_history_without_an_earlier_window_is_refused(span: int) -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="at least one before it"):
        ChannelHistory(span=span)
