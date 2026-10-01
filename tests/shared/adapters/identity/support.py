"""Issuers the token tests sign with: a key pair each, generated once, never fetched."""

import datetime as dt
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from emblema.shared.adapters.identity.jwt_identity_provider import Key

ISSUER = "https://issuer.example"
AUDIENCE = "emblema-api"


class Issuer:
    """An issuer with one signing key, which it publishes under a key identifier."""

    def __init__(self, kid: str) -> None:
        self.kid = kid
        self._private = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    @property
    def private_pem(self) -> bytes:
        return self._private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def resolve(self, token: str) -> Key:
        """The key a token names, as a key set published by this issuer alone would answer.

        Raises:
            jwt.PyJWKClientError: If the token names a key this issuer never published.
        """
        kid = jwt.get_unverified_header(token).get("kid")
        if kid != self.kid:
            raise jwt.PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')
        return self._private.public_key()


SIGNER = Issuer("signer")
STRANGER = Issuer("stranger")


def signed(issuer: Issuer, *, expires_in: float = 60.0, **claims: Any) -> str:
    """A token the issuer signed, valid for a minute unless told otherwise, over the claims."""
    now = dt.datetime.now(dt.UTC)
    stated: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + dt.timedelta(seconds=expires_in),
    }
    stated.update(claims)
    return jwt.encode(stated, issuer.private_pem, algorithm="RS256", headers={"kid": issuer.kid})
