from pathlib import Path

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.exceptions import UnreadablePublishedCorpusError
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockTrainingCorpusReader:
    """Reads a published corpus out of the store: the manifest as JSON, the windows as a block.

    The manifest is decoded by the Catalog's published codec and the block by the shared one, so
    nothing of the Catalog's interior is touched. The block is fetched to the workspace and mapped
    from there, under its digest — the layout the Catalog's archive keeps its own blocks in — so a
    machine that published a corpus reads it back without fetching it again, and any machine
    fetches it once. A block found in the workspace is hashed before it is mapped: the workspace
    is shared with another process and outlives every one of them, and a run over bytes nobody
    verified would sign as a run over the corpus.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        self._store = store
        self._workspace = workspace
        self._json = PublishedCorpusManifestJson()

    def describe(self, manifest: ArtifactRef) -> PretrainingInput:
        published = self._manifest(manifest)
        return PretrainingInput(
            corpus=published.corpus,
            corpus_version=published.corpus_version,
            corpus_checksum=published.corpus_checksum,
            manifest=manifest,
            block_checksum=published.block.checksum,
            vocabulary_size=len(published.channels),
        )

    def read(self, manifest: ArtifactRef, share: CorpusShare) -> TrainingCorpus:
        published = self._manifest(manifest)
        try:
            block = WindowBlock(self._fetched(published.block))
        except MalformedBlockError as error:
            raise UnreadablePublishedCorpusError(
                f"block {published.block.key!r} is not one this reads: {error}"
            ) from error
        positions = {key: index for index, key in enumerate(published.units)}
        return TrainingCorpus(
            name=published.corpus,
            checksum=published.block.checksum,
            training=block.of_units(
                self._positions(positions, share.select(published.training_units))
            ),
            validation=block.of_units(self._positions(positions, published.validation_units)),
            vocabulary_size=len(published.channels),
        )

    @staticmethod
    def _positions(positions: dict[str, int], units: tuple[str, ...]) -> set[int]:
        # A unit that yielded no window is absent from the block and from the manifest's unit
        # order alike; asking for its windows is asking for nothing, not an error.
        return {positions[key] for key in units if key in positions}

    def _manifest(self, ref: ArtifactRef) -> PublishedCorpusManifest:
        try:
            return self._json.decode(self._store.get(ref))
        except MalformedManifestError as error:
            raise UnreadablePublishedCorpusError(
                f"artifact {ref.key!r} is not a published manifest: {error}"
            ) from error

    def _fetched(self, block: ArtifactRef) -> Path:
        path = self._workspace / block.checksum.digest
        if path.is_file() and Checksum.of_chunks(chunks_of(path), block.checksum.algorithm) == (
            block.checksum
        ):
            return path
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._store.get_file(block, path)
        return path
