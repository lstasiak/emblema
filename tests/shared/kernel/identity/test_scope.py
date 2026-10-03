import pytest

from emblema.shared.kernel.exceptions import InvalidScopeError
from emblema.shared.kernel.identity.scope import Scope


@pytest.mark.parametrize(
    "written",
    ["promote", "Serving:Promote", "serving:", ":promote", " serving:promote", "a:b:c", ""],
)
def test_a_scope_is_resource_colon_action_in_lowercase(written: str) -> None:
    with pytest.raises(InvalidScopeError):
        Scope(written)


def test_a_scope_prints_as_written() -> None:
    assert str(Scope("serving:promote")) == "serving:promote"


def test_a_scope_is_compared_as_written() -> None:
    assert Scope("serving:promote") == Scope("serving:promote")
    assert Scope("serving:promote") != Scope("serving:withdraw")
