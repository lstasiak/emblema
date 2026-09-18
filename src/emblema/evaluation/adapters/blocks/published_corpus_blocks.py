from pathlib import Path

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class PublishedCorpusBlocks:
    """The published corpora as this context reads them: manifests decoded, blocks mapped.

    The manifest is decoded by the Catalog's published codec and the block by the shared one, so
    nothing of the Catalog's interior is touched; what either refuses is reported in this
    context's words. One instance serves every adapter of a process that reads blocks, so a block
    is fetched and verified once for all of them rather than once per adapter.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        self._store = store
        self._blocks = BlockWorkspace(store, workspace)
        self._json = PublishedCorpusManifestJson()

    def manifest_of(self, ref: ArtifactRef) -> PublishedCorpusManifest:
        """The manifest stored under ``ref``.

        Raises:
            UnreadableTaskCorpusError: If the artifact is not a published manifest.
        """
        try:
            return self._json.decode(self._store.get(ref))
        except MalformedManifestError as error:
            raise UnreadableTaskCorpusError(
                f"artifact {ref.key!r} is not a published manifest: {error}"
            ) from error

    def block_of(self, published: PublishedCorpusManifest) -> WindowBlock:
        """The block ``published`` names, mapped from the workspace.

        Raises:
            UnreadableTaskCorpusError: If the artifact is not a block this reads.
        """
        try:
            return self._blocks.open(published.block)
        except MalformedBlockError as error:
            raise UnreadableTaskCorpusError(
                f"block {published.block.key!r} is not one this reads: {error}"
            ) from error
