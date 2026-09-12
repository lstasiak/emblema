"""Where the command-line process is assembled: every dependency built once, by hand.

Wiring is written out rather than resolved by a container. At this size a container would hide
the one thing worth reading — which adapter a use case actually got — behind registrations, and
would answer questions about lifetimes that this process does not ask: one command runs, then it
exits. The cost of the choice is that a second process repeats the parts it shares; the moment
that repetition is real rather than hypothetical is the moment to revisit it.

Building the services is separate from running them, and every adapter can be overridden, so a
test can assemble the same process over in-memory adapters and check the wiring itself.
"""

from dataclasses import dataclass
from pathlib import Path

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.readers.cmapss import SUBSETS, CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.corpus_version_ref_assembler import CorpusVersionRefAssembler
from emblema.catalog.application.register_corpus import RegisterCorpus
from emblema.catalog.application.register_corpus_version import RegisterCorpusVersion
from emblema.catalog.application.tokenise_corpus_version import TokeniseCorpusVersion
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.config.settings import Settings
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True)
class Services:
    """The use cases this process can run, each already holding its dependencies."""

    register_corpus: RegisterCorpus
    register_corpus_version: RegisterCorpusVersion
    tokenise_corpus_version: TokeniseCorpusVersion
    archive: CorpusArchive
    corpora: CorpusRepository


def build_services(
    settings: Settings,
    *,
    corpus_root: Path,
    workspace: Path,
    subsets: tuple[str, ...] = SUBSETS,
    corpora: CorpusRepository | None = None,
    reader: CorpusReader | None = None,
    store: ArtifactStore | None = None,
    archive: CorpusArchive | None = None,
    clock: Clock | None = None,
    ids: IdGenerator | None = None,
) -> Services:
    """Assemble the process.

    Args:
        settings: Values read from the environment; only this function and the adapters see them.
        corpus_root: Directory the raw corpus is read from.
        workspace: Directory blocks pass through on their way in or out of the store.
        subsets: Subsets of the corpus to read, where it has any.
        corpora: Repository of corpora; in-memory unless given, because this process registers
            and publishes in one run and nothing yet outlives it.
        reader: Reader of the raw corpus; the C-MAPSS reader unless given.
        store: Artifact store; the configured S3-compatible bucket unless given.
        archive: Corpus archive; a block archive over ``store`` unless given.
        clock: Source of the current instant.
        ids: Source of new identifiers.
    """
    clock = clock or SystemClock()
    ids = ids or Uuid4IdGenerator()
    corpora = corpora or InMemoryCorpusRepository()
    reader = reader or CmapssCorpusReader(corpus_root, subsets)
    if archive is None:
        store = store or _artifact_store(settings)
        archive = BlockCorpusArchive(store, workspace)
    refs = CorpusVersionRefAssembler()
    events = InMemoryEventPublisher(InMemoryEventSubscriber())
    return Services(
        register_corpus=RegisterCorpus(corpora, ids),
        register_corpus_version=RegisterCorpusVersion(corpora, reader, ids, clock, events, refs),
        tokenise_corpus_version=TokeniseCorpusVersion(
            corpora, reader, SlidingWindowTokeniser(), archive
        ),
        archive=archive,
        corpora=corpora,
    )


def _artifact_store(settings: Settings) -> ArtifactStore:
    config = settings.artifact_store
    return S3ArtifactStore.connect(
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key=config.access_key.get_secret_value() if config.access_key else None,
        secret_key=config.secret_key.get_secret_value() if config.secret_key else None,
        bucket=config.bucket,
        key_prefix=config.key_prefix,
    )
