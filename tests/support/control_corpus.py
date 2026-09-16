"""The positive control published the way a real corpus is, cut to a test's size.

Both control layouts are registered, frozen, tokenised, archived and read back through the use
cases and adapters the command line assembles, into one store and under one channel vocabulary.
A test that needs windows of the control asks ``publish_control`` for them rather than building
tokens by hand, so that what reaches the model is what the pipeline produces.
"""

from pathlib import Path
from typing import NamedTuple

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.synthetic.layouts import CONTROL_A, CONTROL_B, CONTROL_PROCESS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.entrypoints.cli.publish_corpus.known_corpora import KnownCorpora
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from tests.support.settings import unreachable_store
from tests.support.synthetic import miniature

# Short enough that a miniature unit yields several windows, with a stride that leaves a tail.
WINDOW = WindowSpec(length=32.0, stride=12.0)


class Control(NamedTuple):
    """Both control corpora published into one store, under one channel vocabulary."""

    archive: CorpusArchive
    manifests: tuple[TokenisationManifest, TokenisationManifest]

    def windows_of(self, which: int) -> list[TokenWindow]:
        manifest = self.manifests[which]
        return list(self.archive.read_windows(manifest.archived, manifest.split.training))

    @property
    def vocabulary_size(self) -> int:
        """Entries of the vocabulary the second corpus continued from the first."""
        return len(self.manifests[1].scheme.vocabulary)


def publish_control(workspace: Path) -> Control:
    store = InMemoryArtifactStore()
    corpora = InMemoryCorpusRepository()
    archive: CorpusArchive | None = None
    refs: list[ArtifactRef] = []
    for layout in (CONTROL_A, CONTROL_B):
        root = process(layout, workspace, store, corpora)
        archive = root.adapters.archive
        # The second corpus continues the first one's vocabulary: that is what lets one model
        # hold both, and the only place the two layouts are ever named together.
        refs.append(root.services.publish_corpus(command(layout, refs[0] if refs else None)))
    if archive is None:
        raise RuntimeError("no layout was published")
    manifests = tuple(archive.read_manifest(ref) for ref in refs)
    return Control(archive, (manifests[0], manifests[1]))


def process(
    layout: SensorLayout,
    workspace: Path,
    store: InMemoryArtifactStore,
    corpora: InMemoryCorpusRepository,
) -> CompositionRoot:
    """The publishing process the command line assembles, over a corpus cut to a test's size."""
    return CompositionRoot(
        unreachable_store(),
        corpus_root=workspace / "raw",
        workspace=workspace,
        corpora=corpora,
        reader=SyntheticCorpusReader(CONTROL_PROCESS, miniature(layout)),
        store=store,
    )


def command(layout: SensorLayout, vocabulary_from: ArtifactRef | None) -> PublishCorpusCommand:
    known = KnownCorpora.default().named(layout.name)
    return PublishCorpusCommand(
        name=known.name,
        source=known.source,
        licence=known.licence,
        window=WINDOW,
        validation_fraction=0.25,
        seed=1,
        vocabulary_from=vocabulary_from,
    )
