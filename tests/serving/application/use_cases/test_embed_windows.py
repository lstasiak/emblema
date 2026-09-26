import pytest

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.serving.adapters.in_memory.inference_runtime import InMemoryInferenceRuntime
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.admission.window_admission import WindowAdmission
from emblema.serving.application.use_cases.embed_windows import EmbedWindows, EmbedWindowsQuery
from emblema.serving.domain.exceptions import (
    EmbeddingUnavailableError,
    NonFiniteAnswerError,
    ServedModelNotServingError,
)
from tests.serving.support import KEPT, MODEL, WITHDRAWN, limits, observed_window, served, stated


def use_case(
    embedding: tuple[float, ...] | None = (0.1, 0.2), *, withdrawn: bool = False
) -> tuple[EmbedWindows, InMemoryInferenceRuntime]:
    runtime = InMemoryInferenceRuntime({KEPT.checksum: stated(embedding=embedding)})
    models = InMemoryServedModelRepository()
    models.save(served().withdraw(WITHDRAWN) if withdrawn else served())
    admission = WindowAdmission(runtime, PublishedWindowTokeniser(), limits())
    return EmbedWindows(models, runtime, admission), runtime


def test_every_window_is_represented_in_the_order_asked() -> None:
    embed, runtime = use_case()

    represented = embed(
        EmbedWindowsQuery(
            served_model=MODEL, windows=(observed_window(), observed_window(length=20.0))
        )
    )

    assert represented.served_model == MODEL
    assert [w.embedding for w in represented.windows] == [(0.1, 0.2), (0.1, 0.2)]
    assert represented.windows[0].warnings == ()
    assert represented.windows[1].warnings == (
        "the window spans 20 and the candidate was fitted on windows of 10",
    )
    assert len(runtime.asked) == 1


def test_a_candidate_without_a_representation_refuses_by_name() -> None:
    embed, _ = use_case(embedding=None)

    with pytest.raises(EmbeddingUnavailableError):
        embed(EmbedWindowsQuery(served_model=MODEL, windows=(observed_window(),)))


def test_a_withdrawn_model_represents_nothing() -> None:
    embed, runtime = use_case(withdrawn=True)

    with pytest.raises(ServedModelNotServingError):
        embed(EmbedWindowsQuery(served_model=MODEL, windows=(observed_window(),)))

    assert runtime.asked == []


def test_a_model_representing_a_window_with_no_number_fails_rather_than_answers() -> None:
    embed, _ = use_case(embedding=(0.1, float("nan")))

    with pytest.raises(NonFiniteAnswerError):
        embed(EmbedWindowsQuery(served_model=MODEL, windows=(observed_window(),)))
