from uuid import UUID

from emblema.shared.kernel.identifiers import EntityId


class SequentialIdGenerator:
    """Generator that hands out identifiers 1, 2, 3, ... so a test can predict every id."""

    def __init__(self) -> None:
        self._next = 1

    def generate[I: EntityId](self, kind: type[I]) -> I:
        identifier = kind(UUID(int=self._next))
        self._next += 1
        return identifier
