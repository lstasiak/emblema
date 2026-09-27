from uuid import UUID

import pytest

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.serving.adapters.in_memory.inference_runtime import InMemoryInferenceRuntime
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.application.use_cases.predict_windows import (
    PredictWindows,
    PredictWindowsQuery,
)
from emblema.serving.domain.exceptions import (
    NonFiniteAnswerError,
    ServedModelNotFoundError,
    ServedModelNotServingError,
)
from emblema.serving.domain.identifiers import ServedModelId
from tests.serving.support import KEPT, MODEL, WITHDRAWN, limits, observed_window, served, stated


def use_case(
    runtime: InMemoryInferenceRuntime | None = None, *, withdrawn: bool = False
) -> tuple[PredictWindows, InMemoryInferenceRuntime]:
    chosen = InMemoryInferenceRuntime({KEPT.checksum: stated()}) if runtime is None else runtime
    models = InMemoryServedModelRepository()
    models.save(served().withdraw(WITHDRAWN) if withdrawn else served())
    admission = WindowAdmission(chosen, PublishedWindowTokeniser(), limits())
    return PredictWindows(models, chosen, admission), chosen


def test_every_window_is_answered_in_the_order_asked_over_the_channels_it_stands_on() -> None:
    predict, runtime = use_case()
    with_unknown = observed_window(
        start=50.0,
        observations=(
            *[
                ObservedValue(channel=o.channel, time=o.time + 50.0, value=o.value)
                for o in observed_window().observations
            ],
            ObservedValue(channel="vibration", time=53.0, value=1.0),
        ),
    )

    answered = predict(
        PredictWindowsQuery(served_model=MODEL, windows=(observed_window(), with_unknown))
    )

    assert answered.served_model == MODEL
    assert [w.prediction for w in answered.windows] == [42.0, 42.0]
    assert answered.windows[0].channels_ignored == ()
    assert answered.windows[0].warnings == ()
    assert answered.windows[1].channels_used == ("age", "pressure", "temperature")
    assert answered.windows[1].channels_ignored == ("vibration",)
    assert answered.windows[1].warnings == (
        "readings on channels the model does not know were ignored: vibration",
    )
    (asked,) = runtime.asked
    assert asked[0] == KEPT
    assert len(asked[1]) == 2


def test_the_runtime_is_asked_once_for_the_whole_request() -> None:
    predict, runtime = use_case()

    predict(PredictWindowsQuery(served_model=MODEL, windows=(observed_window(),) * 3))

    assert len(runtime.asked) == 1


def test_a_model_nobody_stored_is_not_found() -> None:
    predict, _ = use_case()

    with pytest.raises(ServedModelNotFoundError):
        predict(
            PredictWindowsQuery(
                served_model=ServedModelId(UUID(int=99)), windows=(observed_window(),)
            )
        )


def test_a_withdrawn_model_answers_nothing() -> None:
    predict, runtime = use_case(withdrawn=True)

    with pytest.raises(ServedModelNotServingError):
        predict(PredictWindowsQuery(served_model=MODEL, windows=(observed_window(),)))

    assert runtime.asked == []


@pytest.mark.parametrize("answer", [float("nan"), float("inf")])
def test_a_model_answering_with_no_number_fails_rather_than_answers(answer: float) -> None:
    predict, _ = use_case(InMemoryInferenceRuntime({KEPT.checksum: stated(prediction=answer)}))

    with pytest.raises(NonFiniteAnswerError):
        predict(PredictWindowsQuery(served_model=MODEL, windows=(observed_window(),)))
