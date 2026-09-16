from collections.abc import Collection
from pathlib import Path

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockCorpusWindows:
    """Reads a published corpus for a task: the manifest as JSON, the windows out of the block.

    Only the placement of a window is read, never its tokens: a task counts and labels windows,
    and the block holds where each one sits beside the tokens it does not need. The block is
    still fetched whole, because that is how it travels, and hashed before it is mapped when the
    workspace already holds it — bytes nobody verified would label a task after data nobody
    checked.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        self._store = store
        self._workspace = workspace
        self._json = PublishedCorpusManifestJson()

    def describe(self, manifest: ArtifactRef) -> CorpusSides:
        published = self._manifest(manifest)
        return CorpusSides(
            corpus=published.corpus,
            training=frozenset(UnitKey(key) for key in published.training_units),
            validation=frozenset(UnitKey(key) for key in published.validation_units),
        )

    def windows_of(
        self, manifest: ArtifactRef, units: Collection[UnitKey]
    ) -> tuple[TaskWindow, ...]:
        published = self._manifest(manifest)
        named = {UnitKey(key) for key in published.units} | {
            UnitKey(key) for key in published.empty_units
        }
        unknown = sorted(str(unit) for unit in set(units) - named)
        if unknown:
            raise UnreadableTaskCorpusError(
                f"corpus {published.corpus!r} does not name units: {unknown}"
            )
        positions = {key: index for index, key in enumerate(published.units)}
        wanted = {positions[str(unit)] for unit in units if str(unit) in positions}
        block = self._block(published)
        by_position = {index: key for key, index in positions.items()}
        return tuple(
            TaskWindow(
                unit=UnitKey(by_position[block.unit_of(index)]),
                position=index,
                ends_at=block.extent_of(index)[1],
            )
            for index in range(len(block))
            if block.unit_of(index) in wanted
        )

    def _block(self, published: PublishedCorpusManifest) -> WindowBlock:
        try:
            return WindowBlock(self._fetched(published.block))
        except MalformedBlockError as error:
            raise UnreadableTaskCorpusError(
                f"block {published.block.key!r} is not one this reads: {error}"
            ) from error

    def _manifest(self, ref: ArtifactRef) -> PublishedCorpusManifest:
        try:
            return self._json.decode(self._store.get(ref))
        except MalformedManifestError as error:
            raise UnreadableTaskCorpusError(
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
