"""Stand-in for a downstream context that consumes the Catalog's published language.

It imports ``emblema.catalog.contracts`` and ``emblema.shared`` and nothing else of the Catalog;
``tests/architecture/test_context_isolation.py`` proves that in a fresh interpreter.
"""

from emblema.catalog.contracts.corpus_version_ref import CorpusVersionRef
from emblema.catalog.contracts.events import CorpusVersionFrozen
from emblema.shared.ports.event_subscriber import EventSubscriber


class AvailableCorpusVersions:
    """Projection a downstream context keeps of the versions it may train or evaluate on."""

    def __init__(self) -> None:
        self.versions: list[CorpusVersionRef] = []

    def register(self, events: EventSubscriber) -> None:
        events.subscribe(CorpusVersionFrozen, self.on_version_frozen)

    def on_version_frozen(self, event: CorpusVersionFrozen) -> None:
        self.versions.append(event.version)
