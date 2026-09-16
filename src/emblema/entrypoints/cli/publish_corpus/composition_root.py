from pathlib import Path
from typing import Self

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.persistence.corpus_repository import SqlAlchemyCorpusRepository
from emblema.catalog.adapters.readers.cmapss import SUBSETS as CMAPSS_SUBSETS
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
from emblema.catalog.adapters.readers.skab import SUBSETS as SKAB_SUBSETS
from emblema.catalog.adapters.readers.skab import SkabCorpusReader
from emblema.catalog.adapters.readers.smd import SUBSETS as SMD_SUBSETS
from emblema.catalog.adapters.readers.smd import SmdCorpusReader
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
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
from emblema.entrypoints.cli.configured import configured_engine, configured_store, settings_for
from emblema.entrypoints.cli.publish_corpus.adapters import Adapters
from emblema.entrypoints.cli.publish_corpus.services import Services
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
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
        settings: Settings | None = None,
        *,
        corpus: str | None = None,
        corpus_root: Path,
        workspace: Path,
        subsets: tuple[str, ...] = (),
        corpora: CorpusRepository | None = None,
        reader: CorpusReader | None = None,
        store: ArtifactStore | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> None:
        """Assemble the process.

        Args:
            settings: Values read from the environment; only the root and the adapters see them.
                Needed only to build the store or the repository; a process bringing both of its
                own is assembled by ``over`` instead, which requires them.
            corpus: Name of the corpus to publish; it decides which adapter reads it, and is
                needed only where ``reader`` is left to the root to choose.
            corpus_root: Directory the raw corpus is read from, where it is read from files.
            workspace: Directory blocks pass through on their way in or out of the store.
            subsets: Subsets of the corpus to read; all of them where empty.
            corpora: Repository of corpora; the configured metadata database unless given.
            reader: Reader of the raw corpus; the adapter of the named corpus unless given.
                One of the two has to be stated.
            store: Artifact store; the configured S3-compatible bucket unless given. The
                archive is always the block archive over this store.
            clock: Source of the current instant; the system clock unless given.
            ids: Source of new identifiers; random UUIDs unless given.

        Raises:
            ValueError: If the store or the repository is left to the root without settings to
                build it from, or the reader is left to it without a corpus it has an adapter for.
        """
        chosen_store = configured_store(settings_for(settings, "store")) if store is None else store
        self.adapters = Adapters(
            corpora=(
                self._corpus_repository(settings_for(settings, "corpora"))
                if corpora is None
                else corpora
            ),
            reader=self._corpus_reader(corpus, corpus_root, subsets) if reader is None else reader,
            store=chosen_store,
            archive=BlockCorpusArchive(chosen_store, workspace, PublishedCorpusManifestAssembler()),
            clock=SystemClock() if clock is None else clock,
            ids=Uuid4IdGenerator() if ids is None else ids,
        )
        self.services = self._services(self.adapters)

    @classmethod
    def over(
        cls,
        *,
        corpora: CorpusRepository,
        store: ArtifactStore,
        corpus: str | None = None,
        corpus_root: Path,
        workspace: Path,
        subsets: tuple[str, ...] = (),
        reader: CorpusReader | None = None,
        clock: Clock | None = None,
        ids: IdGenerator | None = None,
    ) -> Self:
        """The process over a repository and a store of its own, nothing read from the environment.

        Those two are the only adapters the root builds from settings, and here they are required:
        what the general constructor refuses at runtime when settings are missing, this refuses at
        the type. The reader is chosen as always, by name or by being given.
        """
        return cls(
            corpus=corpus,
            corpus_root=corpus_root,
            workspace=workspace,
            subsets=subsets,
            corpora=corpora,
            reader=reader,
            store=store,
            clock=clock,
            ids=ids,
        )

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
    def _corpus_reader(corpus: str | None, root: Path, subsets: tuple[str, ...]) -> CorpusReader:
        """Which adapter reads which corpus, and what it takes from the process to do it.

        A corpus read from files is handed the directory it was downloaded to; a generated one is
        handed it too and ignores it, because its specification is its data. The root is passed
        either way rather than made conditional here: which corpora have files on disk is the
        adapters' business, and a caller that had to know would be choosing the adapter itself.

        Raises:
            ValueError: If no corpus was named, or none of that name has an adapter.
        """
        match corpus:
            case None:
                raise ValueError("the process needs a corpus to read or a reader to read it with")
            case "cmapss":
                return CmapssCorpusReader(root, subsets or CMAPSS_SUBSETS)
            case "skab":
                return SkabCorpusReader(root, subsets or SKAB_SUBSETS)
            case "smd":
                return SmdCorpusReader(root, subsets or SMD_SUBSETS)
            case "esa_ad":
                return EsaAdCorpusReader(root, subsets or EsaAdCorpusReader.SUBSETS)
            case generated if generated in LAYOUTS:
                return SyntheticCorpusReader(CONTROL_PROCESS, LAYOUTS[generated])
            case _:
                raise ValueError(f"no adapter reads a corpus named {corpus!r}")

    @staticmethod
    def _corpus_repository(settings: Settings) -> CorpusRepository:
        return SqlAlchemyCorpusRepository(configured_engine(settings))
