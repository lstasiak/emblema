from pathlib import Path

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.exceptions import UnreadablePublishedCorpusError
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockTrainingCorpusReader:
    """Reads a published corpus out of the store: the manifest as JSON, the windows as a block.

    The manifest is decoded by the Catalog's published codec and the block by the shared one, so
    nothing of the Catalog's interior is touched. The block is fetched to the workspace and mapped
    from there, under its digest — the layout the Catalog's archive keeps its own blocks in — so a
    machine that published a corpus reads it back without fetching it again.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        self._store = store
        self._blocks = BlockWorkspace(store, workspace)
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
            block = self._blocks.open(published.block)
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
            channels=tuple(
                self.channel_name(channel.corpus, channel.channel) for channel in published.channels
            ),
        )

    @staticmethod
    def channel_name(corpus: str, channel: str) -> str:
        """What a training corpus calls a published channel: its corpus and itself, joined.

        The name carries the corpus because a vocabulary chained through several publications
        may hold one channel name under two corpora, and the mixture compares names.
        """
        return f"{corpus}/{channel}"

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
