from pathlib import Path

from sqlalchemy import create_engine

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.persistence.corpus_repository import SqlAlchemyCorpusRepository
from emblema.catalog.adapters.readers.cmapss import SUBSETS, CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.assemblers.corpus_version_ref_assembler import (
    CorpusVersionRefAssembler,
)
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpus
from emblema.catalog.application.use_cases.register_corpus import RegisterCorpus
from emblema.catalog.application.use_cases.register_corpus_version import RegisterCorpusVersion
from emblema.catalog.application.use_cases.tokenise_corpus_version import TokeniseCorpusVersion
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.config.settings import Settings
from emblema.entrypoints.cli.adapters import Adapters
from emblema.entrypoints.cli.services import Services
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


class CompositionRoot:
    """Assembles the command-line process: every adapter chosen once, every use case built on them.

    Wiring is written out rather than resolved by a container, so the adapter a use case got can
    be read here. Every adapter can be overridden and the ones chosen are exposed, so a test
    assembles the same process over in-memory adapters and checks the wiring itself.

    Attributes:
        adapters: The port implementations the process runs on.
        services: The use cases, each already holding its dependencies.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        corpus_root: Path,
        workspace: Path,
        subsets: tuple[str, ...] = SUBSETS,
        corpora: CorpusRepository | None = None,
        reader: CorpusReader | None = None,
        store: ArtifactStore | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
            corpus_root: Directory the raw corpus is read from.
            workspace: Directory blocks pass through on their way in or out of the store.
            subsets: Subsets of the corpus to read, where it has any.
            corpora: Repository of corpora; the configured metadata database unless given.
            reader: Reader of the raw corpus; the C-MAPSS reader unless given.
            store: Artifact store; the configured S3-compatible bucket unless given. The
                archive is always the block archive over this store.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.
        """
        chosen_store = self._artifact_store(settings) if store is None else store
        self.adapters = Adapters(
            corpora=self._corpus_repository(settings) if corpora is None else corpora,
            reader=CmapssCorpusReader(corpus_root, subsets) if reader is None else reader,
            store=chosen_store,
            archive=BlockCorpusArchive(chosen_store, workspace, PublishedCorpusManifestAssembler()),
            clock=SystemClock() if clock is None else clock,
            ids=Uuid4IdGenerator() if ids is None else ids,
        )
        self.services = self._services(self.adapters)

    @staticmethod
    def _services(adapters: Adapters) -> Services:
        register_corpus = RegisterCorpus(adapters.corpora, adapters.ids)
        register_corpus_version = RegisterCorpusVersion(
            adapters.corpora,
            adapters.reader,
            adapters.ids,
            adapters.clock,
            InMemoryEventPublisher(InMemoryEventSubscriber()),
            CorpusVersionRefAssembler(),
        )
        tokenise_corpus_version = TokeniseCorpusVersion(
            adapters.corpora, adapters.reader, SlidingWindowTokeniser(), adapters.archive
        )
        return Services(
            register_corpus=register_corpus,
            register_corpus_version=register_corpus_version,
            tokenise_corpus_version=tokenise_corpus_version,
            publish_corpus=PublishCorpus(
                register_corpus,
                register_corpus_version,
                tokenise_corpus_version,
                adapters.corpora,
                adapters.reader,
                adapters.archive,
            ),
        )

    @staticmethod
    def _corpus_repository(settings: Settings) -> CorpusRepository:
        # The engine opens no connection until the first query: assembling costs no network.
        return SqlAlchemyCorpusRepository(create_engine(settings.database.sqlalchemy_url()))

    @staticmethod
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
