"""The edge's identification of a caller, over a router that depends on it."""

import pytest

pytest.importorskip("fastapi")

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from emblema.shared.adapters.identity.static_token_identity_provider import (
    StaticTokenIdentityProvider,
)
from emblema.shared.api.bearer_authentication import BearerAuthentication
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.ports.exceptions import IdentityProviderUnavailableError

OPERATOR = Principal(subject="operator")


class Unavailable:
    def identify(self, credential: str) -> Principal:
        raise IdentityProviderUnavailableError("the issuer's keys could not be fetched")


def application(authentication: BearerAuthentication) -> FastAPI:
    protected = APIRouter(dependencies=[Depends(authentication)])

    def whoami(request: Request) -> dict[str, str]:
        return {"subject": authentication.read_actor(request).subject}

    protected.get("/whoami")(whoami)
    open_router = APIRouter()

    def unguarded(request: Request) -> dict[str, str]:
        return {"subject": authentication.read_actor(request).subject}

    open_router.get("/unguarded")(unguarded)
    app = FastAPI()
    app.include_router(protected)
    app.include_router(open_router)
    return app


@pytest.fixture
def client() -> TestClient:
    authentication = BearerAuthentication(StaticTokenIdentityProvider({"t-op": OPERATOR}))
    return TestClient(application(authentication))


def test_a_request_without_a_token_is_challenged(client: TestClient) -> None:
    answered = client.get("/whoami")

    assert answered.status_code == 401
    assert answered.headers["www-authenticate"] == "Bearer"
    assert "bearer token is required" in answered.json()["detail"]


@pytest.mark.parametrize(
    "authorization", ["Bearer t-nobody", "Basic dDpvcA==", "Bearer", "t-op", "Bearer t-op extra"]
)
def test_a_token_that_stands_for_nobody_is_refused_without_saying_why(
    client: TestClient, authorization: str
) -> None:
    answered = client.get("/whoami", headers={"Authorization": authorization})

    assert answered.status_code == 401
    assert answered.headers["www-authenticate"] == "Bearer"


def test_an_identified_caller_is_known_to_the_handler(client: TestClient) -> None:
    answered = client.get("/whoami", headers={"Authorization": "Bearer t-op"})

    assert answered.status_code == 200
    assert answered.json() == {"subject": "operator"}


def test_a_route_reading_the_actor_without_being_protected_is_a_wiring_mistake(
    client: TestClient,
) -> None:
    with pytest.raises(LookupError, match="not protected"):
        client.get("/unguarded", headers={"Authorization": "Bearer t-op"})


def test_a_provider_that_cannot_be_asked_is_not_a_refusal_of_the_caller() -> None:
    client = TestClient(application(BearerAuthentication(Unavailable())))

    with pytest.raises(IdentityProviderUnavailableError):
        client.get("/whoami", headers={"Authorization": "Bearer t-op"})


def test_the_schema_names_the_scheme_once_and_marks_the_protected_route(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["components"]["securitySchemes"] == {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "A token the configured identity provider issued, carrying the scopes it grants."
            ),
        }
    }
    assert schema["paths"]["/whoami"]["get"]["security"] == [{"bearerAuth": []}]
    assert "security" not in schema["paths"]["/unguarded"]["get"]
