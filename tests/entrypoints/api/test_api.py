"""Contract of the HTTP API over in-memory adapters: shapes, statuses and the schema.

Nothing of any model or campaign is decided here; what is held is the surface a client sees —
every answer's shape, every refusal's status and format, the pages and their cursors, the probes
and the metrics — over adapters that answer as they are told.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.entrypoints.api.composition_root import CompositionRoot
from emblema.entrypoints.api.emblema_api import EmblemaApi
from emblema.entrypoints.api.problem_details import ProblemDetails
from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.evaluation.adapters.in_memory.campaign_listing import InMemoryCampaignListing
from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.adapters.in_memory.verdict_memo import InMemoryVerdictMemo
from emblema.serving.adapters.in_memory.inference_runtime import InMemoryInferenceRuntime
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.adapters.in_memory.served_model_listing import InMemoryServedModelListing
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.domain.exceptions import InferenceBusyError
from emblema.serving.domain.identifiers import ServedModelId
from emblema.shared.adapters.identity.static_token_identity_provider import (
    StaticTokenIdentityProvider,
)
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope
from emblema.shared.kernel.tokens import TokenWindow
from tests.evaluation.support import CAMPAIGN, closed_reading, store
from tests.serving.support import (
    FITTED,
    KEPT,
    MODEL,
    OPERATOR,
    OTHER_CAMPAIGN,
    PROMOTED,
    WITHDRAWN,
    limits,
    origin,
    promotable,
    served,
    stated,
)
from tests.support.settings import API, TELEMETRY

MEDIA_TYPE = ProblemDetails.MEDIA_TYPE
WITHDRAWN_MODEL = ServedModelId(UUID(int=15))
ORIGIN = "http://localhost:5173"
PROMOTER = Principal(subject="promoter", scopes=frozenset({Scope("serving:promote")}))
TOKENS = {"t-operator": OPERATOR, "t-promoter": PROMOTER}
AS_OPERATOR = {"Authorization": "Bearer t-operator"}
# What a second finished campaign kept: in the store, promotable, not yet served.
UNSERVED = b"kept by another campaign"


def failing() -> None:
    raise ConnectionError("no route to host")


class _Failing(InMemoryInferenceRuntime):
    """A runtime that fails in a way no layer names, as a graph that cannot run would."""

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        raise RuntimeError("the session could not run: /srv/secret/graph.onnx")


class _Busy(InMemoryInferenceRuntime):
    """A runtime whose networks are running all the budget allows."""

    def predict(self, artifact: ArtifactRef, windows: Sequence[TokenWindow]) -> tuple[float, ...]:
        raise InferenceBusyError("the networks are running as much as they may")

    def embed(
        self, artifact: ArtifactRef, windows: Sequence[TokenWindow]
    ) -> tuple[tuple[float, ...], ...]:
        raise InferenceBusyError("the networks are running as much as they may")


@contextmanager
def client(
    *,
    telemetry: bool = False,
    ready: bool = True,
    origins: str = ORIGIN,
    runtime: InMemoryInferenceRuntime | None = None,
    **settings: Any,
) -> Iterator[TestClient]:
    models = InMemoryServedModelRepository()
    models.save(served())
    models.save(served(served_model_id=WITHDRAWN_MODEL, artifact=KEPT).withdraw(WITHDRAWN))
    campaigns = InMemoryEvaluationCampaignRepository()
    store(campaigns, closed_reading())
    # What finished campaigns kept, bytes in the store: the artifact the served model answers
    # with, and one nothing serves yet.
    artifacts = InMemoryArtifactStore()
    promotables = InMemoryPromotableArtifactRepository()
    promotables.save(promotable(artifact=artifacts.put(FITTED)))
    promotables.save(
        promotable(artifact=artifacts.put(UNSERVED), origin=origin(campaign=OTHER_CAMPAIGN))
    )
    reporting = Telemetry(TELEMETRY) if telemetry else None
    root = CompositionRoot(
        telemetry=reporting,
        limits=limits(),
        store=artifacts,
        identity=StaticTokenIdentityProvider(TOKENS),
        served=models,
        served_listing=InMemoryServedModelListing(models),
        promotables=promotables,
        campaigns=campaigns,
        campaign_listing=InMemoryCampaignListing(campaigns),
        verdicts=InMemoryVerdictMemo(capacity=4),
        runtime=(
            InMemoryInferenceRuntime(
                {KEPT.checksum: stated(), Checksum.of_bytes(UNSERVED): stated()}
            )
            if runtime is None
            else runtime
        ),
        tokeniser=PublishedWindowTokeniser(),
        clock=FixedClock(PROMOTED),
        ids=SequentialIdGenerator(),
        checks={"artifact_store": lambda: None, "database": (lambda: None) if ready else failing},
    )
    api = EmblemaApi(
        root.services,
        root.readiness,
        API.model_copy(update={"cors_origins": origins, **settings}),
        reporting,
        identity=root.adapters.identity,
    )
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def api() -> Iterator[TestClient]:
    with client() as test_client:
        yield test_client


def request_body(**overrides: Any) -> dict[str, Any]:
    window: dict[str, Any] = {
        "start": 0.0,
        "length": 10.0,
        "observations": [
            {"channel": "temperature", "time": 1.0, "value": 0.5},
            {"channel": "pressure", "time": 2.0, "value": 0.75},
        ],
        "static_features": [{"channel": "age", "value": 65.0}],
    }
    return {"windows": [window | overrides]}


def test_the_schema_describes_every_route_by_a_stable_operation(api: TestClient) -> None:
    schema = api.get("/openapi.json").json()

    operations = {
        (path, method): operation["operationId"]
        for path, methods in schema["paths"].items()
        for method, operation in methods.items()
    }
    assert operations == {
        ("/health", "get"): "health",
        ("/ready", "get"): "ready",
        ("/served-models", "get"): "list_served_models",
        ("/served-models/{served_model_id}", "get"): "view_served_model",
        ("/served-models/{served_model_id}/predictions", "post"): "predict",
        ("/served-models/{served_model_id}/embeddings", "post"): "embed",
        ("/served-models", "post"): "promote",
        ("/served-models/{served_model_id}/withdrawal", "post"): "withdraw",
        ("/campaigns", "get"): "list_campaigns",
        ("/campaigns/{campaign_id}", "get"): "view_campaign",
        ("/campaigns/{campaign_id}/runs", "get"): "list_campaign_runs",
    }
    assert "Problem" in schema["components"]["schemas"]


def test_the_schema_marks_exactly_the_routes_that_change_what_is_served_as_protected(
    api: TestClient,
) -> None:
    schema = api.get("/openapi.json").json()

    protected = {
        (path, method)
        for path, methods in schema["paths"].items()
        for method, operation in methods.items()
        if operation.get("security") == [{"bearerAuth": []}]
    }
    assert protected == {
        ("/served-models", "post"),
        ("/served-models/{served_model_id}/withdrawal", "post"),
    }
    assert list(schema["components"]["securitySchemes"]) == ["bearerAuth"]


def test_the_process_reports_up_and_ready(api: TestClient) -> None:
    assert api.get("/health").json() == {"status": "ok"}
    ready = api.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "checks": {"artifact_store": "ok", "database": "ok"}}


def test_a_dependency_that_does_not_answer_makes_the_process_not_ready() -> None:
    with client(ready=False) as api:
        ready = api.get("/ready")

        assert ready.status_code == 503
        assert ready.json()["status"] == "not ready"
        # Named, and nothing more: what an unreachable service says about itself stays in the log.
        assert ready.json()["checks"] == {"artifact_store": "ok", "database": "failed"}
        assert "no route to host" not in ready.text


def test_a_served_model_answers_windows_naming_what_each_answer_stands_on(
    api: TestClient,
) -> None:
    body = request_body()
    body["windows"].append(
        {
            "start": 50.0,
            "length": 20.0,
            "observations": [
                {"channel": "temperature", "time": 51.0, "value": 0.5},
                {"channel": "vibration", "time": 52.0, "value": 9.0},
            ],
        }
    )

    answered = api.post(f"/served-models/{MODEL}/predictions", json=body)

    assert answered.status_code == 200
    assert answered.json() == {
        "served_model_id": str(MODEL),
        "windows": [
            {
                "prediction": 42.0,
                "channels_used": ["age", "pressure", "temperature"],
                "channels_ignored": [],
                "warnings": [],
            },
            {
                "prediction": 42.0,
                "channels_used": ["temperature"],
                "channels_ignored": ["vibration"],
                "warnings": [
                    "readings on channels the model does not know were ignored: vibration",
                    "the window spans 20 and the candidate was fitted on windows of 10",
                ],
            },
        ],
    }


def test_a_served_model_represents_windows(api: TestClient) -> None:
    represented = api.post(f"/served-models/{MODEL}/embeddings", json=request_body())

    assert represented.status_code == 200
    assert represented.json()["windows"][0]["embedding"] == [0.1, 0.2]


def test_a_served_model_is_shown_with_what_it_takes(api: TestClient) -> None:
    shown = api.get(f"/served-models/{MODEL}")

    assert shown.status_code == 200
    assert shown.json()["model"]["served_model_id"] == str(MODEL)
    assert shown.json()["model"]["state"] == "serving"
    assert shown.json()["input"]["corpus"] == "test-corpus"
    assert [c["name"] for c in shown.json()["input"]["channels"] if not c["known"]] == ["silent"]


def test_served_models_are_paged_by_cursor(api: TestClient) -> None:
    first = api.get("/served-models").json()
    second = api.get("/served-models", params={"cursor": first["next_cursor"]}).json()

    # Both were promoted at the same instant, so identity breaks the tie.
    assert [m["served_model_id"] for m in first["items"]] == [str(MODEL)]
    assert [m["served_model_id"] for m in second["items"]] == [str(WITHDRAWN_MODEL)]
    assert second["next_cursor"] is None
    assert [
        m["state"]
        for m in api.get("/served-models", params={"state": "serving", "limit": 5}).json()["items"]
    ] == ["serving"]


def test_a_campaign_is_listed_shown_and_its_runs_paged(api: TestClient) -> None:
    listed = api.get("/campaigns").json()
    shown = api.get(f"/campaigns/{CAMPAIGN}").json()
    runs = api.get(f"/campaigns/{CAMPAIGN}/runs", params={"limit": 2}).json()
    rest = api.get(
        f"/campaigns/{CAMPAIGN}/runs", params={"limit": 2, "cursor": runs["next_cursor"]}
    ).json()

    assert [c["campaign_id"] for c in listed["items"]] == [str(CAMPAIGN)]
    assert shown["campaign"]["finished"] is True
    assert shown["verdict"]["sentence"] == closed_reading().verdict().sentence()
    assert len(shown["curves"]) == 2
    assert len(runs["items"]) == 2
    assert len(rest["items"]) == 2
    assert runs["items"] != rest["items"]


@pytest.mark.parametrize(
    ("method", "path", "body", "status", "detail"),
    [
        ("get", f"/served-models/{UUID(int=99)}", None, 404, "no served model"),
        ("get", "/served-models/not-a-uuid", None, 422, "not a UUID"),
        ("post", f"/served-models/{WITHDRAWN_MODEL}/predictions", request_body(), 409, "withdrawn"),
        ("post", f"/served-models/{MODEL}/predictions", {"windows": []}, 422, "windows"),
        ("post", f"/served-models/{MODEL}/predictions", request_body(length=-1.0), 422, "length"),
        (
            "post",
            f"/served-models/{MODEL}/predictions",
            request_body(observations=[{"channel": "vibration", "time": 1.0, "value": 1.0}]),
            422,
            "vibration",
        ),
        ("get", "/served-models", {"cursor": "nonsense"}, 422, "cursor"),
        ("get", f"/campaigns/{UUID(int=99)}", None, 404, "no campaign"),
        ("get", "/nowhere", None, 404, "Not Found"),
    ],
)
def test_every_refusal_answers_as_a_problem_with_its_status_and_reason(
    api: TestClient,
    method: str,
    path: str,
    body: dict[str, Any] | None,
    status: int,
    detail: str,
) -> None:
    answered = api.post(path, json=body) if method == "post" else api.get(path, params=body)

    assert answered.status_code == status
    assert answered.headers["content-type"] == MEDIA_TYPE
    problem = answered.json()
    assert problem["status"] == status
    assert problem["instance"] == path
    assert detail in problem["detail"]


def test_too_many_windows_at_once_are_refused(api: TestClient) -> None:
    body = {"windows": request_body()["windows"] * 5}

    answered = api.post(f"/served-models/{MODEL}/predictions", json=body)

    assert answered.status_code == 422
    assert "at most 4 windows" in answered.json()["detail"]


def test_a_browser_from_a_configured_origin_may_call(api: TestClient) -> None:
    answered = api.get("/health", headers={"Origin": ORIGIN})

    assert answered.headers["access-control-allow-origin"] == ORIGIN


def test_metrics_count_what_was_answered_and_what_was_ignored() -> None:
    body = request_body(
        observations=[
            {"channel": "temperature", "time": 1.0, "value": 0.5},
            {"channel": "vibration", "time": 2.0, "value": 9.0},
        ]
    )
    with client(telemetry=True) as api:
        api.post(f"/served-models/{MODEL}/predictions", json=body)

        exposition = api.get("/metrics")

        assert exposition.status_code == 200
        assert "emblema_windows_answered_total" in exposition.text
        assert "emblema_inference_duration_seconds" in exposition.text
        assert "emblema_tokenisation_duration_seconds" in exposition.text
        assert 'answer="prediction"' in exposition.text
        assert "emblema_channels_ignored_total" in exposition.text


def test_without_configured_origins_no_browser_is_admitted() -> None:
    with client(origins="") as api:
        answered = api.get("/health", headers={"Origin": ORIGIN})

        assert "access-control-allow-origin" not in answered.headers


def test_a_failure_no_layer_named_answers_as_a_problem_that_hides_its_cause() -> None:
    with client(runtime=_Failing({KEPT.checksum: stated()})) as api:
        answered = api.post(f"/served-models/{MODEL}/predictions", json=request_body())

        assert answered.status_code == 500
        assert answered.headers["content-type"] == MEDIA_TYPE
        assert answered.json()["detail"].startswith("the service failed to answer")
        assert "secret" not in answered.text


def test_a_failure_names_the_trace_it_is_logged_under_where_requests_are_traced() -> None:
    with client(telemetry=True, runtime=_Failing({KEPT.checksum: stated()})) as api:
        answered = api.post(f"/served-models/{MODEL}/predictions", json=request_body())

        assert answered.status_code == 500
        assert "logged under trace " in answered.json()["detail"]


def test_a_model_answering_with_no_number_is_a_failure_of_the_service() -> None:
    with client(runtime=InMemoryInferenceRuntime({KEPT.checksum: stated(float("nan"))})) as api:
        answered = api.post(f"/served-models/{MODEL}/predictions", json=request_body())

        assert answered.status_code == 500
        assert answered.headers["content-type"] == MEDIA_TYPE


def test_a_body_larger_than_the_service_reads_is_refused_unread() -> None:
    with client(max_request_bytes=64) as api:
        answered = api.post(f"/served-models/{MODEL}/predictions", json=request_body())

        assert answered.status_code == 413
        assert answered.headers["content-type"] == MEDIA_TYPE
        assert "at most 64 bytes" in answered.json()["detail"]


def test_a_body_sent_without_its_length_is_refused(api: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"windows": []}'

    answered = api.post(
        f"/served-models/{MODEL}/predictions",
        content=chunks(),
        headers={"content-type": "application/json"},
    )

    assert answered.status_code == 411
    assert answered.headers["content-type"] == MEDIA_TYPE


@pytest.mark.parametrize("route", ["predictions", "embeddings"])
def test_a_service_running_all_it_may_asks_the_caller_to_come_back(route: str) -> None:
    with client(runtime=_Busy({KEPT.checksum: stated()})) as api:
        answered = api.post(f"/served-models/{MODEL}/{route}", json=request_body())

        assert answered.status_code == 503
        assert answered.headers["content-type"] == MEDIA_TYPE
        assert answered.headers["retry-after"] == str(API.retry_after_seconds())
        assert "running as much as they may" in answered.json()["detail"]


def test_a_refusal_because_the_service_is_busy_is_counted_apart_from_a_failure() -> None:
    with client(telemetry=True, runtime=_Busy({KEPT.checksum: stated()})) as api:
        api.post(f"/served-models/{MODEL}/predictions", json=request_body())

        exposition = api.get("/metrics").text

        assert 'emblema_inference_refused_total{answer="prediction"' in exposition


def test_a_kept_artifact_is_promoted_with_a_token_and_withdrawn_with_it(api: TestClient) -> None:
    kept = Checksum.of_bytes(UNSERVED)

    promoted = api.post("/served-models", json={"checksum": str(kept)}, headers=AS_OPERATOR)

    assert promoted.status_code == 201
    model = promoted.json()["served_model_id"]
    assert promoted.headers["location"] == f"/served-models/{model}"
    assert api.get(f"/served-models/{model}").json()["model"]["state"] == "serving"

    withdrawn = api.post(f"/served-models/{model}/withdrawal", headers=AS_OPERATOR)

    assert withdrawn.status_code == 204
    assert withdrawn.content == b""
    assert api.get(f"/served-models/{model}").json()["model"]["state"] == "withdrawn"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/served-models", {"checksum": str(KEPT.checksum)}),
        (f"/served-models/{MODEL}/withdrawal", None),
    ],
)
def test_a_change_to_what_is_served_without_a_token_is_challenged(
    api: TestClient, path: str, body: dict[str, Any] | None
) -> None:
    answered = api.post(path, json=body)

    assert answered.status_code == 401
    assert answered.headers["content-type"] == MEDIA_TYPE
    assert answered.headers["www-authenticate"] == "Bearer"
    assert answered.json()["detail"] == "a bearer token is required"
    assert api.get(f"/served-models/{MODEL}").json()["model"]["state"] == "serving"


def test_a_token_that_stands_for_nobody_is_refused_the_same_way(api: TestClient) -> None:
    answered = api.post(
        "/served-models",
        json={"checksum": str(KEPT.checksum)},
        headers={"Authorization": "Bearer t-nobody"},
    )

    assert answered.status_code == 401
    assert answered.headers["www-authenticate"] == "Bearer"


def test_a_caller_identified_but_not_granted_the_operation_is_forbidden(api: TestClient) -> None:
    withdrawn = api.post(
        f"/served-models/{MODEL}/withdrawal", headers={"Authorization": "Bearer t-promoter"}
    )

    assert withdrawn.status_code == 403
    assert withdrawn.headers["content-type"] == MEDIA_TYPE
    assert "serving:withdraw" in withdrawn.json()["detail"]
    assert api.get(f"/served-models/{MODEL}").json()["model"]["state"] == "serving"


@pytest.mark.parametrize(
    ("path", "body", "status", "detail"),
    [
        ("/served-models", {"checksum": str(KEPT.checksum)}, 409, "already"),
        ("/served-models", {"checksum": f"sha256:{'0' * 64}"}, 404, "no finished campaign"),
        ("/served-models", {"checksum": "not-a-checksum"}, 422, "checksum"),
        ("/served-models", {"checksum": str(KEPT.checksum), "campaign_id": "x"}, 422, "campaign"),
        ("/served-models", {"checksum": str(KEPT.checksum), "candidate": " "}, 422, "candidate"),
        (f"/served-models/{WITHDRAWN_MODEL}/withdrawal", None, 409, "already been withdrawn"),
        (f"/served-models/{UUID(int=99)}/withdrawal", None, 404, "no served model"),
    ],
)
def test_a_refused_change_answers_as_a_problem_with_its_status_and_reason(
    api: TestClient, path: str, body: dict[str, Any] | None, status: int, detail: str
) -> None:
    answered = api.post(path, json=body, headers=AS_OPERATOR)

    assert answered.status_code == status
    assert answered.headers["content-type"] == MEDIA_TYPE
    assert detail in answered.json()["detail"]


@pytest.mark.parametrize("route", ["predictions", "embeddings"])
def test_a_caller_past_its_allowance_on_the_open_routes_is_told_when_to_come_back(
    route: str,
) -> None:
    with client(inference_requests_per_minute=2) as api:
        for _ in range(2):
            assert (
                api.post(f"/served-models/{MODEL}/{route}", json=request_body()).status_code == 200
            )

        refused = api.post(f"/served-models/{MODEL}/{route}", json=request_body())

        assert refused.status_code == 429
        assert refused.headers["content-type"] == MEDIA_TYPE
        assert refused.headers["retry-after"] == "30"
        assert "2 requests a minute" in refused.json()["detail"]
        # Reading is not rationed: the same caller still sees what is served.
        assert api.get(f"/served-models/{MODEL}").status_code == 200


def test_the_two_open_routes_share_one_allowance() -> None:
    with client(inference_requests_per_minute=1) as api:
        assert (
            api.post(f"/served-models/{MODEL}/predictions", json=request_body()).status_code == 200
        )

        assert (
            api.post(f"/served-models/{MODEL}/embeddings", json=request_body()).status_code == 429
        )
