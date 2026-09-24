from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class ServedModelId(EntityId):
    """Identity of a served model.

    Not published: no other context refers to what Serving serves, and a request that names a
    model reaches it through this context's own entrypoints.
    """
