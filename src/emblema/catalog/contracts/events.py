from dataclasses import dataclass

from emblema.catalog.contracts.corpus_version_ref import CorpusVersionRef
from emblema.shared.events.domain_event import DomainEvent


@dataclass(frozen=True, kw_only=True)
class CorpusVersionFrozen(DomainEvent):
    """A corpus version became immutable and may now be referenced by other contexts.

    Attributes:
        version: Published reference to the frozen version.
    """

    version: CorpusVersionRef
