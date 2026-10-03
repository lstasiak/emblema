import pytest

from emblema.shared.kernel.exceptions import InvalidPrincipalError
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope

PROMOTE = Scope("serving:promote")
WITHDRAW = Scope("serving:withdraw")


@pytest.mark.parametrize("subject", ["", " ", " operator", "operator\n"])
def test_a_subject_is_non_empty_unpadded_text(subject: str) -> None:
    with pytest.raises(InvalidPrincipalError):
        Principal(subject=subject)


def test_a_principal_permits_exactly_the_scopes_it_was_granted() -> None:
    principal = Principal(subject="operator", scopes=frozenset({PROMOTE}))

    assert principal.permits(PROMOTE)
    assert not principal.permits(WITHDRAW)


def test_a_principal_is_granted_nothing_by_default() -> None:
    assert not Principal(subject="visitor").permits(PROMOTE)
