from typing import Self

from pydantic import BaseModel, Field, SecretStr, model_validator

from emblema.config.named_issuer import NamedIssuer
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope


class IdentitySettings(BaseModel):
    """Who the HTTP process trusts to say who a caller is, read as ``EMBLEMA_IDENTITY__<FIELD>``.

    One provider or the other: tokens this process was given, for a local stack and a CI run,
    or an issuer whose signed tokens it verifies by the keys the issuer publishes. Naming both
    is refused, as is naming neither: a process with routes to protect does not guess.
    """

    static_tokens: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Tokens the process accepts as they are, comma-separated, each written "
            "token:subject:scopes with the scopes space-separated as a token's scope claim "
            "writes them; the token itself holds no colon. Kept as one string and split by the "
            "process, so that a list survives every shell and file it is written in."
        ),
    )
    issuer: str | None = Field(
        default=None, description="The issuer whose tokens are accepted, as its tokens name it."
    )
    audience: str | None = Field(
        default=None, description="Whom a token must be issued to for this process to accept it."
    )
    jwks_url: str | None = Field(
        default=None, description="Where the issuer publishes the keys its tokens are signed with."
    )

    @model_validator(mode="after")
    def _one_provider(self) -> Self:
        parts = (("issuer", self.issuer), ("audience", self.audience), ("jwks_url", self.jwks_url))
        named = [name for name, value in parts if value]
        if named and len(named) < len(parts):
            raise ValueError(
                "an issuer is named by all of issuer, audience and jwks_url, "
                f"got only {', '.join(named)}"
            )
        if bool(named) == bool(self.static_tokens.get_secret_value().strip()):
            raise ValueError("name either static tokens or an issuer, not both and not neither")
        return self

    def named_issuer(self) -> NamedIssuer | None:
        """The issuer whose tokens are accepted, or ``None`` where static tokens are."""
        if self.issuer is None or self.audience is None or self.jwks_url is None:
            return None
        return NamedIssuer(issuer=self.issuer, audience=self.audience, jwks_url=self.jwks_url)

    def principals(self) -> dict[str, Principal]:
        """The static tokens and whom each stands for.

        Raises:
            ValueError: If an entry is not written token:subject:scopes, a scope is not one, or
                one token is listed twice.
        """
        principals: dict[str, Principal] = {}
        for entry in self.static_tokens.get_secret_value().split(","):
            if not entry.strip():
                continue
            token, subject, scopes = self._parts(entry)
            if token in principals:
                raise ValueError("a static token is listed once; one is listed twice")
            principals[token] = Principal(
                subject=subject, scopes=frozenset(Scope(word) for word in scopes.split())
            )
        return principals

    @staticmethod
    def _parts(entry: str) -> tuple[str, str, str]:
        parts = entry.strip().split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"a static token is written token:subject:scopes, got {entry!r}")
        token, subject, scopes = parts
        return token.strip(), subject.strip(), scopes
