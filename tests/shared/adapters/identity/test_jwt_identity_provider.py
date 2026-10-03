"""What the token adapter refuses beyond the port's contract, and how it reads a token's claims."""

import datetime as dt
from typing import Any

import pytest

pytest.importorskip("jwt")

import jwt

from emblema.shared.adapters.identity.jwt_identity_provider import JwtIdentityProvider
from emblema.shared.kernel.identity.scope import Scope
from emblema.shared.ports.exceptions import (
    CredentialRejectedError,
    IdentityProviderUnavailableError,
)
from tests.shared.adapters.identity.support import AUDIENCE, ISSUER, SIGNER, signed


def provider() -> JwtIdentityProvider:
    return JwtIdentityProvider(issuer=ISSUER, audience=AUDIENCE, keys=SIGNER.resolve)


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        ({"sub": "operator", "expires_in": -1.0}, "expired"),
        ({"sub": "operator", "iss": "https://someone.else"}, "another issuer"),
        ({"sub": "operator", "aud": "another-service"}, "another audience"),
        ({"scope": "serving:promote"}, "no subject"),
        ({"sub": "operator", "scope": "Promote"}, "a scope not written as one"),
        ({"sub": "operator", "scp": ["serving:"]}, "a listed scope not written as one"),
        ({"sub": "operator", "scp": ["serving:promote", 7]}, "a listed scope that is not text"),
        ({"sub": " "}, "a blank subject"),
    ],
)
def test_a_token_is_rejected_for(claims: dict[str, Any], reason: str) -> None:
    with pytest.raises(CredentialRejectedError):
        provider().identify(signed(SIGNER, **claims))


def test_a_token_without_expiry_is_rejected() -> None:
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "operator", "iat": dt.datetime.now(dt.UTC)},
        SIGNER.private_pem,
        algorithm="RS256",
        headers={"kid": SIGNER.kid},
    )

    with pytest.raises(CredentialRejectedError):
        provider().identify(token)


def test_a_token_that_is_not_one_is_rejected() -> None:
    with pytest.raises(CredentialRejectedError):
        provider().identify("not.a.token")


def test_a_token_signed_with_a_shared_secret_is_rejected_whatever_it_claims() -> None:
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "operator", "exp": 2**31},
        "a-shared-secret-of-thirty-two-bytes!",
        algorithm="HS256",
        headers={"kid": SIGNER.kid},
    )

    with pytest.raises(CredentialRejectedError):
        provider().identify(token)


def test_scopes_are_read_from_the_space_separated_claim_and_the_listed_one() -> None:
    token = signed(SIGNER, sub="operator", scope="serving:promote", scp=["serving:withdraw"])

    principal = provider().identify(token)

    assert principal.scopes == {Scope("serving:promote"), Scope("serving:withdraw")}


def test_a_token_with_no_scope_claim_names_a_principal_granted_nothing() -> None:
    principal = provider().identify(signed(SIGNER, sub="reader"))

    assert principal.subject == "reader"
    assert principal.scopes == frozenset()


def test_a_little_clock_skew_is_forgiven_only_when_asked() -> None:
    token = signed(SIGNER, sub="operator", expires_in=-5.0)

    with pytest.raises(CredentialRejectedError):
        provider().identify(token)
    lenient = JwtIdentityProvider(
        issuer=ISSUER, audience=AUDIENCE, keys=SIGNER.resolve, leeway_seconds=30.0
    )
    assert lenient.identify(token).subject == "operator"


def test_an_unreachable_key_set_is_reported_as_unavailable_not_as_a_refusal() -> None:
    def unreachable(token: str) -> bytes:
        raise jwt.PyJWKClientConnectionError("Fail to fetch data from the url")

    unavailable = JwtIdentityProvider(issuer=ISSUER, audience=AUDIENCE, keys=unreachable)

    with pytest.raises(IdentityProviderUnavailableError):
        unavailable.identify(signed(SIGNER, sub="operator"))


def test_over_a_published_key_set_the_key_is_the_one_the_token_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    class Published:
        key = SIGNER.resolve(signed(SIGNER, sub="operator"))

    def get_signing_key_from_jwt(self: jwt.PyJWKClient, token: str) -> Published:
        asked.append(token)
        return Published()

    monkeypatch.setattr(jwt.PyJWKClient, "get_signing_key_from_jwt", get_signing_key_from_jwt)
    over = JwtIdentityProvider.over_jwks(
        issuer=ISSUER, audience=AUDIENCE, jwks_url="https://issuer.example/jwks"
    )
    token = signed(SIGNER, sub="operator")

    assert over.identify(token).subject == "operator"
    assert asked == [token]
