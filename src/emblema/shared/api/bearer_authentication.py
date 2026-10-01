from http import HTTPStatus
from typing import Annotated, ClassVar

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.ports.exceptions import CredentialRejectedError
from emblema.shared.ports.identity_provider import IdentityProvider


class BearerAuthentication:
    """Turns the bearer token of a request into the principal it stands for, or refuses.

    A router that protects its routes depends on this once, so no handler of it runs for an
    unidentified caller; a handler that needs to know who called reads the principal back off
    the request. Identifying is the edge's business and ends here: what the principal may do is
    asked by the use case, of the principal, in the context's own words.

    Attributes:
        SCHEME: The scheme as the schema documents it, under one name every protected route
            shares. It does not refuse on its own: a missing header is answered here, in the
            one error shape.
    """

    SCHEME: ClassVar[HTTPBearer] = HTTPBearer(
        scheme_name="bearerAuth",
        auto_error=False,
        description=(
            "A token the configured identity provider issued, carrying the scopes it grants."
        ),
    )
    _STATE: ClassVar[str] = "principal"

    def __init__(self, identity: IdentityProvider) -> None:
        self._identity = identity

    def __call__(
        self,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(SCHEME)],
    ) -> Principal:
        """Identify the caller and remember who they are for the rest of the request.

        Raises:
            HTTPException: With ``401`` if no bearer token was sent or it stands for nobody; the
                challenge header names the scheme, as the status requires.
            IdentityProviderUnavailableError: If the provider could not be asked; the process
                answers that as its own unavailability, not as the caller's fault.
        """
        if credentials is None:
            raise self._unauthorised("a bearer token is required")
        try:
            principal = self._identity.identify(credentials.credentials)
        except CredentialRejectedError as refusal:
            raise self._unauthorised("the bearer token was not accepted") from refusal
        setattr(request.state, self._STATE, principal)
        return principal

    @classmethod
    def read_actor(cls, request: Request) -> Principal:
        """The principal identified for this request.

        Raises:
            LookupError: If the request was not identified, which means a route reads the actor
                without being behind this authentication: a wiring mistake, not a refusal.
        """
        principal = getattr(request.state, cls._STATE, None)
        if not isinstance(principal, Principal):
            raise LookupError("the request was not identified: the route is not protected")
        return principal

    @staticmethod
    def _unauthorised(detail: str) -> HTTPException:
        return HTTPException(
            HTTPStatus.UNAUTHORIZED.value, detail=detail, headers={"WWW-Authenticate": "Bearer"}
        )
