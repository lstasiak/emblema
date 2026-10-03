import hmac
from collections.abc import Mapping

from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.ports.exceptions import CredentialRejectedError


class StaticTokenIdentityProvider:
    """Recognises a fixed set of tokens, each standing for one principal.

    The provider of a local stack, a CI run and a test: whoever holds a token is the principal
    it was issued to, and nothing expires. Tokens are compared in constant time, so a wrong one
    is not told how much of it was right.
    """

    def __init__(self, principals: Mapping[str, Principal]) -> None:
        """Hold the tokens and whom each stands for.

        Raises:
            ValueError: If a token is blank or padded; such a token would be unusable in a
                header and is a configuration mistake, not a credential.
        """
        for token in principals:
            if not token or token != token.strip():
                raise ValueError("a static token is non-empty, unpadded text")
        self._principals = dict(principals)

    def identify(self, credential: str) -> Principal:
        presented = credential.encode()
        for token, principal in self._principals.items():
            if hmac.compare_digest(token.encode(), presented):
                return principal
        raise CredentialRejectedError("the token is not one this service was given")
