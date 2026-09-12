"""An event published by the Catalog reaches a downstream context through its contracts only.

This module plays the Catalog: it freezes a version with the domain model, assembles the
published reference and publishes the event. The subscriber lives in ``downstream`` and knows
nothing of the Catalog beyond ``emblema.catalog.contracts``.
"""

from uuid import UUID

from emblema.catalog.application.assemblers.corpus_version_ref_assembler import (
    CorpusVersionRefAssembler,
)
from emblema.catalog.contracts.events import CorpusVersionFrozen
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.events.domain_event import EventId
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import AT, LICENCE, SCHEMA, content, empty_corpus, version_id
from tests.contracts.downstream import AvailableCorpusVersions


def test_frozen_version_reaches_the_downstream_projection() -> None:
    subscriptions = InMemoryEventSubscriber()
    publisher = InMemoryEventPublisher(subscriptions)
    downstream = AvailableCorpusVersions()
    downstream.register(subscriptions)
    corpus = (
        empty_corpus()
        .add_version(version_id(), SCHEMA, SamplingRegime.REGULAR, LICENCE)
        .record_content(version_id(), content())
        .freeze_version(version_id(), AT)
    )
    reference = CorpusVersionRefAssembler().assemble(corpus.get_version(version_id()))

    publisher.publish(
        CorpusVersionFrozen(event_id=EventId(UUID(int=1)), occurred_at=AT, version=reference)
    )

    assert downstream.versions == [reference]
