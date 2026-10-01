"""Contract of the IdentityProvider port, run against every adapter.

Each adapter is given the same two principals and asked for them by whatever credential it
recognises; a credential of nobody's is refused the same way by all of them.
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest

# The static adapter needs nothing; the token adapter needs the library the API extra brings.
pytest.importorskip("jwt")

from emblema.shared.adapters.identity.jwt_identity_provider import JwtIdentityProvider
from emblema.shared.adapters.identity.static_token_identity_provider import (
    StaticTokenIdentityProvider,
)
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope
from emblema.shared.ports.exceptions import CredentialRejectedError
from emblema.shared.ports.identity_provider import IdentityProvider
from tests.shared.adapters.identity.support import AUDIENCE, ISSUER, SIGNER, STRANGER, signed

OPERATOR = Principal(
    subject="operator", scopes=frozenset({Scope("serving:promote"), Scope("serving:withdraw")})
)
READER = Principal(subject="reader")


class Harness(NamedTuple):
    """A provider under test and a credential of each principal it was given."""

    provider: IdentityProvider
    operator: str
    reader: str
    stranger: str


def static_token() -> Harness:
    return Harness(
        StaticTokenIdentityProvider({"token-of-operator": OPERATOR, "token-of-reader": READER}),
        operator="token-of-operator",
        reader="token-of-reader",
        stranger="token-of-nobody",
    )


def jwt() -> Harness:
    provider = JwtIdentityProvider(issuer=ISSUER, audience=AUDIENCE, keys=SIGNER.resolve)
    return Harness(
        provider,
        operator=signed(SIGNER, sub="operator", scope="serving:promote serving:withdraw"),
        reader=signed(SIGNER, sub="reader"),
        stranger=signed(STRANGER, sub="operator", scope="serving:promote"),
    )


ADAPTERS: dict[str, Callable[[], Harness]] = {"static_token": static_token, "jwt": jwt}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def harness(request: pytest.FixtureRequest) -> Harness:
    build: Callable[[], Harness] = request.param
    return build()


def test_a_credential_of_nobody_is_rejected(harness: Harness) -> None:
    with pytest.raises(CredentialRejectedError):
        harness.provider.identify(harness.stranger)


def test_an_empty_credential_is_rejected(harness: Harness) -> None:
    with pytest.raises(CredentialRejectedError):
        harness.provider.identify("")


def test_a_credential_names_its_principal_and_what_it_was_granted(harness: Harness) -> None:
    assert harness.provider.identify(harness.operator) == OPERATOR


def test_two_credentials_are_told_apart(harness: Harness) -> None:
    assert harness.provider.identify(harness.reader) == READER
    assert harness.provider.identify(harness.reader) != harness.provider.identify(harness.operator)
