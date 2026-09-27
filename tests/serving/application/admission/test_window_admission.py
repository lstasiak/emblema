from collections.abc import Sequence

import pytest

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.serving.adapters.in_memory.inference_runtime import InMemoryInferenceRuntime
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.domain.exceptions import (
    EmptyRequestError,
    TooManyWindowsError,
    UnobservedWindowError,
    UntokenisableRequestError,
    WindowTooLongError,
)
from emblema.shared.kernel.tokens import TokenWindow
from tests.serving.support import KEPT, limits, observed_window, served, stated


def admission(**overrides: int) -> WindowAdmission:
    runtime = InMemoryInferenceRuntime({KEPT.checksum: stated()})
    return WindowAdmission(runtime, PublishedWindowTokeniser(), limits(**overrides))


def test_a_window_on_known_channels_is_tokenised_without_a_warning() -> None:
    (prepared,) = admission().prepare(served(), [observed_window()])

    assert len(prepared.tokens) == 4
    assert prepared.admitted.ignored == ()
    assert prepared.warnings == ()


def test_readings_on_unknown_channels_are_dropped_with_a_warning_naming_them() -> None:
    window = observed_window(
        observations=(
            *observed_window().observations,
            ObservedValue(channel="vibration", time=3.0, value=1.0),
        )
    )

    (prepared,) = admission().prepare(served(), [window])

    assert len(prepared.tokens) == 4
    assert prepared.admitted.ignored == ("vibration",)
    assert prepared.warnings == (
        "readings on channels the model does not know were ignored: vibration",
    )


def test_a_window_of_another_length_than_the_candidate_was_fitted_on_is_remarked_on() -> None:
    (prepared,) = admission().prepare(served(), [observed_window(length=20.0)])

    assert prepared.warnings == (
        "the window spans 20 and the candidate was fitted on windows of 10",
    )


def test_windows_are_prepared_in_the_order_given() -> None:
    first, second = observed_window(), observed_window(length=20.0)

    prepared = admission().prepare(served(), [second, first])

    assert [p.admitted.window for p in prepared] == [second, first]


def test_a_request_without_a_window_is_refused() -> None:
    with pytest.raises(EmptyRequestError):
        admission().prepare(served(), [])


def test_a_request_of_more_windows_than_the_service_admits_is_refused() -> None:
    with pytest.raises(TooManyWindowsError):
        admission(max_windows_per_request=1).prepare(served(), [observed_window()] * 2)


def test_a_window_that_tokenises_to_more_than_the_service_admits_is_refused() -> None:
    with pytest.raises(WindowTooLongError, match="got 4"):
        admission(max_tokens_per_window=3).prepare(served(), [observed_window()])


def test_every_reading_the_model_takes_becomes_one_token() -> None:
    # What refusing a long window before tokenising it rests on.
    window = observed_window()

    (prepared,) = admission().prepare(served(), [window])

    assert len(prepared.tokens) == len(window.observations) + len(window.static_features)


class _Untouchable:
    def tokenise(
        self, window: ObservedWindow, corpus: str, channels: Sequence[PublishedChannel]
    ) -> TokenWindow:
        raise AssertionError("a window refused for its length was tokenised")


def test_a_window_too_long_is_refused_before_any_of_it_is_tokenised() -> None:
    runtime = InMemoryInferenceRuntime({KEPT.checksum: stated()})
    refusing = WindowAdmission(runtime, _Untouchable(), limits(max_tokens_per_window=3))

    with pytest.raises(WindowTooLongError):
        refusing.prepare(served(), [observed_window()])


def test_a_window_with_nothing_the_model_takes_is_refused() -> None:
    window = observed_window(
        observations=(ObservedValue(channel="vibration", time=3.0, value=1.0),)
    )

    with pytest.raises(UnobservedWindowError):
        admission().prepare(served(), [window])


def test_a_timed_reading_on_a_static_channel_is_refused_as_a_bad_request() -> None:
    window = observed_window(
        observations=(
            *observed_window().observations,
            ObservedValue(channel="age", time=3.0, value=61.0),
        )
    )

    with pytest.raises(UntokenisableRequestError, match="timeless"):
        admission().prepare(served(), [window])
