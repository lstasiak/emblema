from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class CorpusId(EntityId):
    """Identity of a corpus, private to the Catalog.

    Other contexts reference versions, never the corpus, so this identifier stays in the domain
    while ``CorpusVersionId`` lives in the published language.
    """
