from dataclasses import dataclass

from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True)
class Adapters:
    """Every port implementation the process runs on, so that what a use case got can be read."""

    corpora: CorpusRepository
    reader: CorpusReader
    store: ArtifactStore
    archive: CorpusArchive
    clock: Clock
    ids: IdGenerator
