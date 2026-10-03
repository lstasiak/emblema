from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self

import jwt
from jwt.algorithms import AllowedPublicKeys

from emblema.shared.kernel.exceptions import InvalidPrincipalError, InvalidScopeError
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope
from emblema.shared.ports.exceptions import (
    CredentialRejectedError,
    IdentityProviderUnavailableError,
)

Key = AllowedPublicKeys | jwt.PyJWK | str | bytes
# The signing key of a credential, looked up by the credential itself, since its header names
# the key. The lookup speaks the library's errors, which are translated here.
KeySource = Callable[[str], Key]


class JwtIdentityProvider:
    """A principal from a signed token of an external issuer, verified without asking it.

    The issuer publishes its signing keys; the token names which one signed it; the signature,
    the issuer, the audience and the expiry are checked here, so a request costs no round trip.
    The subject is the token's ``sub``. The scopes are read from ``scope``, the space-separated
    string OAuth writes, and from ``scp``, the list some issuers write instead; a token may carry
    either or both.

    Attributes:
        ALGORITHMS: The signatures accepted unless told otherwise: asymmetric only, because a
            provider that verifies with a shared secret could also mint, and this adapter exists
            so that the service holds nothing it could mint with.
    """

    ALGORITHMS: ClassVar[tuple[str, ...]] = ("RS256", "RS384", "RS512", "ES256", "ES384", "ES512")

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        keys: KeySource,
        algorithms: Sequence[str] | None = None,
        leeway_seconds: float = 0.0,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._keys = keys
        self._algorithms = tuple(self.ALGORITHMS if algorithms is None else algorithms)
        self._leeway = leeway_seconds

    @classmethod
    def over_jwks(cls, *, issuer: str, audience: str, jwks_url: str) -> Self:
        """The provider over the issuer's published key set, fetched and cached on first use."""
        client = jwt.PyJWKClient(jwks_url)
        return cls(
            issuer=issuer,
            audience=audience,
            keys=lambda token: client.get_signing_key_from_jwt(token).key,
        )

    def identify(self, credential: str) -> Principal:
        try:
            key = self._keys(credential)
        except jwt.PyJWKClientConnectionError as failure:
            raise IdentityProviderUnavailableError(
                "the issuer's signing keys could not be fetched"
            ) from failure
        except jwt.PyJWTError as refusal:
            raise CredentialRejectedError(f"the token names no known key: {refusal}") from refusal
        try:
            claims = jwt.decode(
                credential,
                key,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as refusal:
            raise CredentialRejectedError(f"the token was not accepted: {refusal}") from refusal
        return self._principal(claims)

    @classmethod
    def _principal(cls, claims: dict[str, Any]) -> Principal:
        """The principal the verified claims describe.

        Raises:
            CredentialRejectedError: If the subject or a scope is not written as one.
        """
        try:
            return Principal(
                subject=str(claims["sub"]),
                scopes=frozenset(Scope(word) for word in cls._scope_words(claims)),
            )
        except (InvalidPrincipalError, InvalidScopeError) as refusal:
            raise CredentialRejectedError(
                f"the token's claims are malformed: {refusal}"
            ) from refusal

    @staticmethod
    def _scope_words(claims: dict[str, Any]) -> list[str]:
        """Every scope the claims name, from either claim.

        Raises:
            CredentialRejectedError: If the listed claim holds anything but text.
        """
        words: list[str] = []
        named = claims.get("scope")
        if isinstance(named, str):
            words.extend(named.split())
        listed = claims.get("scp")
        if isinstance(listed, list):
            if not all(isinstance(word, str) for word in listed):
                raise CredentialRejectedError("the token lists a scope that is not text")
            words.extend(listed)
        return words
