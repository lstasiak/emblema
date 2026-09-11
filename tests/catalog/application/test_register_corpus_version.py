from collections.abc import Iterator
from dataclasses import dataclass, field
from uuid import UUID

import pytest

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.application.corpus_version_ref_assembler import CorpusVersionRefAssembler
from emblema.catalog.application.register_corpus import RegisterCorpus, RegisterCorpusCommand
from emblema.catalog.application.register_corpus_version import (
    RegisterCorpusVersion,
    RegisterCorpusVersionCommand,
)
from emblema.catalog.contracts.events import CorpusVersionFrozen
from emblema.catalog.domain.corpus_description import CorpusDescription
from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.exceptions import (
    CorpusNotFoundError,
    CorpusReadError,
    MalformedCorpusDataError,
    SameDataAlreadyFrozenError,
)
from emblema.catalog.domain.identifiers import CorpusId, UnitKey
from emblema.catalog.domain.observation import Observation
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.events.domain_event import EventId
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    SCHEMA,
    SOURCE,
    content,
    corpus_id,
    description,
)


class FailingCorpusReader:
    """A reader whose source is broken: every operation fails the same way."""

    def __init__(self, error: CorpusReadError) -> None:
        self._error = error

    def describe(self) -> CorpusDescription:
        raise self._error

    def read_units(self) -> Iterator[CorpusUnit]:
        raise self._error

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        raise self._error


@dataclass
class Catalog:
    """The Catalog wired on in-memory adapters, with one corpus registered and no version."""

    corpora: InMemoryCorpusRepository = field(default_factory=InMemoryCorpusRepository)
    subscriptions: InMemoryEventSubscriber = field(default_factory=InMemoryEventSubscriber)
    ids: SequentialIdGenerator = field(default_factory=SequentialIdGenerator)
    received: list[CorpusVersionFrozen] = field(default_factory=list)

    def __post_init__(self) -> None:
        register = RegisterCorpus(self.corpora, self.ids)
        self.corpus_id = register(RegisterCorpusCommand(name="C-MAPSS", source=SOURCE))
        self.subscriptions.subscribe(CorpusVersionFrozen, self.received.append)

    def register_version(self, reader: CorpusReader) -> RegisterCorpusVersion:
        return RegisterCorpusVersion(
            self.corpora,
            reader,
            self.ids,
            FixedClock(AT),
            InMemoryEventPublisher(self.subscriptions),
            CorpusVersionRefAssembler(),
        )

    def versions(self) -> tuple[int, ...]:
        return tuple(version.number for version in self.corpora.get(self.corpus_id).versions)


@pytest.fixture
def catalog() -> Catalog:
    return Catalog()


@pytest.fixture
def reader() -> InMemoryCorpusReader:
    return InMemoryCorpusReader(description())


@pytest.fixture
def register(catalog: Catalog, reader: InMemoryCorpusReader) -> RegisterCorpusVersion:
    return catalog.register_version(reader)


def command(corpus_id: CorpusId) -> RegisterCorpusVersionCommand:
    return RegisterCorpusVersionCommand(corpus_id=corpus_id, licence=LICENCE)


def test_registers_what_the_reader_saw_as_a_frozen_version(
    catalog: Catalog, register: RegisterCorpusVersion
) -> None:
    ref = register(command(catalog.corpus_id))

    version = catalog.corpora.get(catalog.corpus_id).get_version(ref.version_id)
    assert version.frozen_at == AT
    assert (version.channel_schema, version.sampling_regime) == (SCHEMA, SamplingRegime.REGULAR)
    assert (version.licence, version.content) == (LICENCE, content())
    assert ref == CorpusVersionRefAssembler().assemble(version)


def test_publishes_the_frozen_version_once_it_is_stored(
    catalog: Catalog, register: RegisterCorpusVersion
) -> None:
    stored_when_received: list[bool] = []
    catalog.subscriptions.subscribe(
        CorpusVersionFrozen,
        lambda event: stored_when_received.append(
            catalog.corpora.get(catalog.corpus_id).get_version(event.version.version_id).is_frozen
        ),
    )

    ref = register(command(catalog.corpus_id))

    assert catalog.received == [
        CorpusVersionFrozen(event_id=EventId(UUID(int=3)), occurred_at=AT, version=ref)
    ]
    assert stored_when_received == [True]


def test_the_same_data_is_not_registered_twice(
    catalog: Catalog, register: RegisterCorpusVersion
) -> None:
    register(command(catalog.corpus_id))

    with pytest.raises(SameDataAlreadyFrozenError):
        register(command(catalog.corpus_id))

    assert catalog.versions() == (1,)
    assert len(catalog.received) == 1


def test_changed_data_becomes_a_new_version_with_another_checksum(
    catalog: Catalog, reader: InMemoryCorpusReader, register: RegisterCorpusVersion
) -> None:
    first = register(command(catalog.corpus_id))

    reader.replace_data(description(b"changed"))
    second = register(command(catalog.corpus_id))

    assert second.version_id != first.version_id
    assert second.checksum != first.checksum
    assert catalog.versions() == (1, 2)
    assert all(version.is_frozen for version in catalog.corpora.get(catalog.corpus_id).versions)


def test_a_failed_reading_leaves_the_corpus_untouched_and_publishes_nothing(
    catalog: Catalog,
) -> None:
    register = catalog.register_version(FailingCorpusReader(MalformedCorpusDataError("bad row")))

    with pytest.raises(MalformedCorpusDataError):
        register(command(catalog.corpus_id))

    assert catalog.versions() == ()
    assert catalog.received == []


def test_an_unknown_corpus_is_rejected_before_the_data_is_read(catalog: Catalog) -> None:
    register = catalog.register_version(FailingCorpusReader(MalformedCorpusDataError("unread")))

    with pytest.raises(CorpusNotFoundError):
        register(command(corpus_id(9)))
