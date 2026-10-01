from dataclasses import dataclass

from emblema.shared.kernel.exceptions import InvalidPrincipalError
from emblema.shared.kernel.identity.scope import Scope


@dataclass(frozen=True, kw_only=True)
class Principal:
    """A caller the edge has identified, carrying the scopes it was granted.

    Who identified the caller and how is the edge's business; what the caller may do is a rule
    of each context, asked of this object by the use case that performs the operation. The
    principal itself decides nothing.

    Invariants: the subject is non-empty text without surrounding whitespace.

    Attributes:
        subject: Whom the credential stood for, as the provider names them.
        scopes: The operations the caller was granted; none by default.
    """

    subject: str
    scopes: frozenset[Scope] = frozenset()

    def __post_init__(self) -> None:
        if not self.subject or self.subject != self.subject.strip():
            raise InvalidPrincipalError(
                f"a subject is non-empty, unpadded text, got {self.subject!r}"
            )

    def permits(self, scope: Scope) -> bool:
        """Whether the caller was granted the scope."""
        return scope in self.scopes
