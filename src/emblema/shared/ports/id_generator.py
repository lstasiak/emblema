from typing import Protocol

from emblema.shared.kernel.identifiers import EntityId


class IdGenerator(Protocol):
    """Source of fresh identifiers of a requested kind, injected so identity stays reproducible."""

    def generate[I: EntityId](self, kind: type[I]) -> I: ...
