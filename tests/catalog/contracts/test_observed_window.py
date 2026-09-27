from dataclasses import replace

import pytest

from emblema.catalog.contracts.exceptions import InvalidObservedWindowError
from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.static_value import StaticValue

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def reading(channel: str = "temperature", time: float = 1.0, value: float = 0.5) -> ObservedValue:
    return ObservedValue(channel=channel, time=time, value=value)


def window(**overrides: object) -> ObservedWindow:
    stated = ObservedWindow(
        start=0.0,
        length=10.0,
        observations=(reading(), reading("pressure", 9.5, 2.0)),
        static_features=(StaticValue(channel="age", value=61.0),),
    )
    return replace(stated, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("channel", ["", "  ", " x", "x "])
def test_a_reading_names_its_channel(channel: str) -> None:
    with pytest.raises(InvalidObservedWindowError):
        reading(channel=channel)
    with pytest.raises(InvalidObservedWindowError):
        StaticValue(channel=channel, value=1.0)


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_reading_is_finite_in_time_and_value(value: float) -> None:
    with pytest.raises(InvalidObservedWindowError):
        reading(time=value)
    with pytest.raises(InvalidObservedWindowError):
        reading(value=value)
    with pytest.raises(InvalidObservedWindowError):
        StaticValue(channel="age", value=value)


@pytest.mark.parametrize("length", [0.0, -1.0, *NON_FINITE])
def test_a_window_has_a_positive_finite_length(length: float) -> None:
    with pytest.raises(InvalidObservedWindowError, match="length"):
        window(length=length)


@pytest.mark.parametrize("start", NON_FINITE)
def test_a_window_starts_at_a_finite_instant(start: float) -> None:
    with pytest.raises(InvalidObservedWindowError, match="start"):
        window(start=start)


def test_a_window_holds_at_least_one_timed_reading() -> None:
    with pytest.raises(InvalidObservedWindowError, match="timed reading"):
        window(observations=())


@pytest.mark.parametrize("time", [-0.5, 10.0, 12.0])
def test_a_reading_lies_inside_the_window(time: float) -> None:
    with pytest.raises(InvalidObservedWindowError, match="outside"):
        window(observations=(reading(time=time),))


def test_a_reading_at_the_start_is_inside_and_one_at_the_end_is_not() -> None:
    assert window(observations=(reading(time=0.0),)).end == 10.0
    with pytest.raises(InvalidObservedWindowError, match="outside"):
        window(observations=(reading(time=10.0),))


def test_a_static_channel_is_stated_once() -> None:
    twice = (StaticValue(channel="age", value=1.0), StaticValue(channel="age", value=2.0))

    with pytest.raises(InvalidObservedWindowError, match="twice"):
        window(static_features=twice)


def test_the_channels_of_a_window_are_those_of_every_reading_timed_or_static() -> None:
    assert window().channels() == frozenset({"temperature", "pressure", "age"})
