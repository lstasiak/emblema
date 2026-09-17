"""The publishing process as one use case, over adapters that reach nothing outside."""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.assemblers.corpus_version_ref_assembler import (
    CorpusVersionRefAssembler,
)
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpus, PublishCorpusCommand
from emblema.catalog.application.use_cases.register_corpus import RegisterCorpus
from emblema.catalog.application.use_cases.register_corpus_version import RegisterCorpusVersion
from emblema.catalog.application.use_cases.tokenise_corpus_version import TokeniseCorpusVersion
from emblema.catalog.domain.channels.channel_schema import ChannelSchema
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.catalog.domain.tokenisation.split_policy import SeededSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.catalog.domain.support import (
    LICENCE,
    OTHER_SCHEMA,
    SCHEMA,
    SOURCE,
    description,
    measured_units,
)

WINDOW = WindowSpec(length=4.0, stride=2.0)


class Registry(NamedTuple):
    """What every publication in one test shares: where corpora and artifacts are kept."""

    corpora: InMemoryCorpusRepository
    store: InMemoryArtifactStore
    archive: BlockCorpusArchive


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    store = InMemoryArtifactStore()
    archive = BlockCorpusArchive(store, tmp_path / "workspace", PublishedCorpusManifestAssembler())
    return Registry(InMemoryCorpusRepository(), store, archive)


def publisher(
    registry: Registry,
    schema: ChannelSchema,
    units: Sequence[tuple[CorpusUnit, list[Observation]]],
    data: bytes = b"records",
) -> PublishCorpus:
    """The use case over a reader of ``units``, sharing the registry with every other publisher."""
    told = description(data, schema, units=len(units), observations=sum(len(o) for _, o in units))
    reader = InMemoryCorpusReader(told, units)
    ids = Uuid4IdGenerator()
    return PublishCorpus(
        RegisterCorpus(registry.corpora, ids),
        RegisterCorpusVersion(
            registry.corpora,
            reader,
            ids,
            SystemClock(),
            InMemoryEventPublisher(InMemoryEventSubscriber()),
            CorpusVersionRefAssembler(),
        ),
        TokeniseCorpusVersion(registry.corpora, reader, SlidingWindowTokeniser(), registry.archive),
        registry.corpora,
        reader,
        registry.archive,
    )


def command(
    name: str = "alpha", vocabulary_from: ArtifactRef | None = None
) -> PublishCorpusCommand:
    return PublishCorpusCommand(
        name=name,
        source=SOURCE,
        licence=LICENCE,
        window=WINDOW,
        split=SeededSplit(0.25, 1),
        vocabulary_from=vocabulary_from,
    )


def registered(registry: Registry, name: str = "alpha") -> Corpus:
    corpus = registry.corpora.find_by_name(name)
    assert corpus is not None
    return corpus


def test_publishing_takes_a_corpus_from_raw_data_to_a_manifest(registry: Registry) -> None:
    publish = publisher(registry, SCHEMA, measured_units(4))

    ref = publish(command())

    manifest = registry.archive.read_manifest(ref)
    assert manifest.corpus == "alpha"
    assert manifest.window == WINDOW
    assert len(manifest.units) == 4
    assert (len(manifest.split.training), len(manifest.split.validation)) == (3, 1)
    assert registry.store.exists(manifest.block)
    assert manifest.corpus_version == registered(registry).versions[0].id


def test_publishing_the_same_data_twice_yields_the_same_manifest(registry: Registry) -> None:
    # The reference to the manifest is what a run pins, so the same data under the same name and
    # configuration must come back as one reference, from one registered version.
    publish = publisher(registry, SCHEMA, measured_units(4))

    first = publish(command())
    second = publish(command())

    assert first == second
    assert len(registered(registry).versions) == 1


def test_changed_data_becomes_a_new_version_of_the_same_corpus(registry: Registry) -> None:
    first = publisher(registry, SCHEMA, measured_units(4))(command())
    again = publisher(registry, SCHEMA, measured_units(4), data=b"records, revised")

    second = again(command())

    corpus = registered(registry)
    assert first != second
    assert [version.number for version in corpus.versions] == [1, 2]
    assert registry.archive.read_manifest(second).corpus_version == corpus.versions[1].id


def test_a_corpus_continues_the_vocabulary_of_the_manifest_it_names(registry: Registry) -> None:
    # Channel identifiers index the model's embedding table, so two corpora a model is trained on
    # together must not both hand out identifier 1: the second grows the first one's vocabulary.
    first = publisher(registry, SCHEMA, measured_units(4))(command("alpha"))
    publish = publisher(registry, OTHER_SCHEMA, measured_units(4, ("vibration",), prefix="m"))

    second = publish(command("beta", vocabulary_from=first))

    vocabulary = registry.archive.read_manifest(second).scheme.vocabulary
    assert [entry.channel_id for entry in vocabulary.entries_of("alpha")] == [1, 2]
    assert vocabulary.id_of("beta", "vibration") == 3
    assert len(registry.archive.read_manifest(first).scheme.vocabulary) == 2


def test_the_channels_of_the_earlier_corpus_are_not_fitted_again(registry: Registry) -> None:
    first = publisher(registry, SCHEMA, measured_units(4))(command("alpha"))
    publish = publisher(registry, OTHER_SCHEMA, measured_units(4, ("vibration",), prefix="m"))

    second = publish(command("beta", vocabulary_from=first))

    scheme = registry.archive.read_manifest(second).scheme
    assert scheme.statistics[0] is None
    assert scheme.statistics_of(scheme.vocabulary.id_of("beta", "vibration")).count == 24


def test_naming_a_manifest_that_was_never_archived_fails_before_anything_is_registered(
    registry: Registry,
) -> None:
    publish = publisher(registry, SCHEMA, measured_units(4))
    missing = content_address(b"never stored", Retention.DURABLE)

    with pytest.raises(ArtifactNotFoundError):
        publish(command(vocabulary_from=missing))

    assert registry.corpora.find_by_name("alpha") is None
