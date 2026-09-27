import pytest

from emblema.serving.domain.admitted_window import AdmittedWindow
from emblema.serving.domain.exceptions import InvalidAdmittedWindowError
from tests.serving.support import observed_window


def test_the_channels_used_are_those_of_the_window_sorted() -> None:
    admitted = AdmittedWindow(window=observed_window(), ignored=("vibration",))

    assert admitted.used == ("age", "pressure", "temperature")


def test_an_ignored_channel_cannot_still_hold_a_reading() -> None:
    with pytest.raises(InvalidAdmittedWindowError, match="still hold"):
        AdmittedWindow(window=observed_window(), ignored=("pressure",))


@pytest.mark.parametrize("ignored", [("b", "a"), ("a", "a")])
def test_ignored_channels_are_named_once_each_in_order(ignored: tuple[str, ...]) -> None:
    with pytest.raises(InvalidAdmittedWindowError, match="once each"):
        AdmittedWindow(window=observed_window(), ignored=ignored)
