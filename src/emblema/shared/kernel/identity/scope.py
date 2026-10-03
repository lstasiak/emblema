import re
from dataclasses import dataclass
from typing import ClassVar

from emblema.shared.kernel.exceptions import InvalidScopeError


@dataclass(frozen=True)
class Scope:
    """One operation a caller may be granted, named as ``resource:action``.

    Invariants: lowercase, one colon, a word on each side. Compared as written, so a scope an
    issuer grants matches only the exact name a policy asks for.

    Attributes:
        value: The scope as a token writes it.
    """

    # `resource:action` as OAuth deployments commonly write a scope: the colon is the one
    # separator and each side is a lowercase word.
    _PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_-]*:[a-z][a-z0-9_-]*$")

    value: str

    def __post_init__(self) -> None:
        if not self._PATTERN.fullmatch(self.value):
            raise InvalidScopeError(
                f"a scope is written resource:action in lowercase, got {self.value!r}"
            )

    def __str__(self) -> str:
        return self.value
