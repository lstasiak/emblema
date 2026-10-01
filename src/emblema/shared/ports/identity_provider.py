from typing import Protocol

from emblema.shared.kernel.identity.principal import Principal


class IdentityProvider(Protocol):
    """Tells who a credential stands for, and what they were granted.

    The one port the driving side consumes: authenticating a caller is the edge's technical
    concern, so the HTTP layer asks here and passes the principal inwards, and no use case sees
    a credential. Whether the principal may do what it asks is each context's own rule.
    """

    def identify(self, credential: str) -> Principal:
        """The principal the credential stands for.

        Raises:
            CredentialRejectedError: If the credential stands for nobody this provider knows.
            IdentityProviderUnavailableError: If the provider could not be asked.
        """
        ...
