from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class CorpusVersionId(EntityId):
    """Identity of a corpus version, the one Catalog entity other contexts refer to.

    It lives in the published language rather than in the domain because both must agree on it:
    the domain imports it from here, and a consumer names the type without touching the domain.
    """
