from uuid import uuid4

from emblema.shared.kernel.identifiers import EntityId


class Uuid4IdGenerator:
    def generate[I: EntityId](self, kind: type[I]) -> I:
        return kind(uuid4())
