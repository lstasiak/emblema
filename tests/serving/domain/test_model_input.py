from dataclasses import replace

import pytest

from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.static_value import StaticValue
from emblema.serving.domain.exceptions import InvalidModelInputError, UnobservedWindowError
from tests.serving.support import CHANNELS, model_input, observed_window


@pytest.mark.parametrize("corpus", ["", " ", " x"])
def test_a_model_input_names_its_corpus(corpus: str) -> None:
    with pytest.raises(InvalidModelInputError, match="corpus"):
        model_input(corpus=corpus)


@pytest.mark.parametrize("length", [0.0, -1.0, float("inf"), float("nan")])
def test_a_model_input_has_a_positive_finite_window_length(length: float) -> None:
    with pytest.raises(InvalidModelInputError, match="window length"):
        model_input(window_length=length)


def test_a_model_takes_at_least_one_channel() -> None:
    with pytest.raises(InvalidModelInputError, match="at least one channel"):
        model_input(channels=())


def test_a_channel_is_named_once() -> None:
    with pytest.raises(InvalidModelInputError, match="twice"):
        model_input(channels=(CHANNELS[0], replace(CHANNELS[0], channel_id=2)))


def test_a_model_knows_the_channels_training_fitted_and_not_the_others() -> None:
    assert model_input().knows("temperature")
    assert model_input().knows("age")
    assert not model_input().knows("silent")
    assert not model_input().knows("vibration")


def test_a_window_on_known_channels_is_admitted_whole() -> None:
    admitted = model_input().admit(observed_window())

    assert admitted.window == observed_window()
    assert admitted.ignored == ()
    assert admitted.used == ("age", "pressure", "temperature")


def test_readings_on_channels_the_model_does_not_know_are_dropped_and_named() -> None:
    window = observed_window(
        observations=(
            *observed_window().observations,
            ObservedValue(channel="vibration", time=3.0, value=1.0),
            ObservedValue(channel="silent", time=4.0, value=1.0),
        ),
        static_features=(
            *observed_window().static_features,
            StaticValue(channel="height", value=1.8),
        ),
    )

    admitted = model_input().admit(window)

    assert admitted.window == observed_window()
    assert admitted.ignored == ("height", "silent", "vibration")


def test_a_window_left_without_a_timed_reading_is_refused_naming_what_was_dropped() -> None:
    window = observed_window(
        observations=(ObservedValue(channel="vibration", time=3.0, value=1.0),)
    )

    with pytest.raises(UnobservedWindowError, match="vibration"):
        model_input().admit(window)


def test_a_window_with_only_static_readings_left_is_refused() -> None:
    window = observed_window(observations=(ObservedValue(channel="silent", time=3.0, value=1.0),))

    with pytest.raises(UnobservedWindowError):
        model_input().admit(window)
