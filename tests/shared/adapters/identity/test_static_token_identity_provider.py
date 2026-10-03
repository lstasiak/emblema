import pytest

from emblema.shared.adapters.identity.static_token_identity_provider import (
    StaticTokenIdentityProvider,
)
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.ports.exceptions import CredentialRejectedError

OPERATOR = Principal(subject="operator")


@pytest.mark.parametrize("token", ["", " ", " token", "token "])
def test_a_blank_or_padded_token_is_refused_when_the_provider_is_built(token: str) -> None:
    with pytest.raises(ValueError, match="unpadded"):
        StaticTokenIdentityProvider({token: OPERATOR})


@pytest.mark.parametrize(
    "presented", ["token-of-operato", "token-of-operator ", "Token-of-operator"]
)
def test_only_the_whole_token_as_issued_is_the_token(presented: str) -> None:
    provider = StaticTokenIdentityProvider({"token-of-operator": OPERATOR})

    with pytest.raises(CredentialRejectedError):
        provider.identify(presented)


def test_a_provider_given_no_tokens_recognises_nobody() -> None:
    with pytest.raises(CredentialRejectedError):
        StaticTokenIdentityProvider({}).identify("anything")
