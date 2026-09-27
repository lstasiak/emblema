import pytest

from emblema.serving.adapters.in_memory.inference_runtime import InMemoryInferenceRuntime
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.read_models.served_model_summary import ServedModelSummary
from emblema.serving.application.use_cases.view_served_model import (
    ViewServedModel,
    ViewServedModelQuery,
)
from emblema.serving.domain.exceptions import ArtifactUnavailableError, ServedModelNotFoundError
from emblema.serving.domain.served_model_state import ServedModelState
from tests.serving.support import KEPT, MODEL, WITHDRAWN, served, stated


def use_case(*, kept: bool = True, withdrawn: bool = False) -> ViewServedModel:
    runtime = InMemoryInferenceRuntime({KEPT.checksum: stated()} if kept else {})
    models = InMemoryServedModelRepository()
    models.save(served().withdraw(WITHDRAWN) if withdrawn else served())
    return ViewServedModel(models, runtime)


def test_a_served_model_is_shown_with_what_it_takes() -> None:
    view = use_case()(ViewServedModelQuery(served_model=MODEL))

    assert view.summary == ServedModelSummary.of(served())
    assert view.summary.state is ServedModelState.SERVING
    assert view.input.corpus == "test-corpus"
    assert view.input.window_length == 10.0
    assert [(c.name, c.timeless, c.known) for c in view.input.channels] == [
        ("temperature", False, True),
        ("pressure", False, True),
        ("age", True, True),
        ("silent", False, False),
    ]


def test_a_withdrawn_model_is_still_shown() -> None:
    view = use_case(withdrawn=True)(ViewServedModelQuery(served_model=MODEL))

    assert view.summary.state is ServedModelState.WITHDRAWN
    assert view.summary.withdrawn_at == WITHDRAWN


def test_a_model_nobody_stored_is_not_found() -> None:
    query = ViewServedModelQuery(served_model=served().served_model_id)
    models = InMemoryServedModelRepository()

    with pytest.raises(ServedModelNotFoundError):
        ViewServedModel(models, InMemoryInferenceRuntime({}))(query)


def test_a_model_whose_artifact_is_gone_says_so() -> None:
    with pytest.raises(ArtifactUnavailableError):
        use_case(kept=False)(ViewServedModelQuery(served_model=MODEL))
