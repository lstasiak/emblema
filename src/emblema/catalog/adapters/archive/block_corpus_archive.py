import os
import tempfile
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.catalog.domain.exceptions import (
    UnreadableCorpusBlockError,
    WindowNotArchivableError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.adapters.windows.exceptions import MalformedBlockError, UnstorableWindowError
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockCorpusArchive:
    """A corpus archived as a block of windows beside a JSON manifest, both in an artifact store.

    The block never passes through memory: it is written in the workspace a window at a time,
    handed to the store as a file and fetched back as a file so that a reader can map it. The
    workspace keeps every block this machine wrote or fetched under its digest, so a corpus is
    downloaded once.
    """

    def __init__(
        self, store: ArtifactStore, workspace: Path, manifests: PublishedCorpusManifestAssembler
    ) -> None:
        self._store = store
        self._workspace = workspace
        self._blocks = BlockWorkspace(store, workspace)
        self._manifests = manifests
        self._json = PublishedCorpusManifestJson()

    def write_windows(self, windows: Iterable[PlacedWindow]) -> ArchivedCorpus:
        self._workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._workspace) as scratch:
            path = Path(scratch) / "corpus.block"
            units, window_count, token_count = self._write_block(path, windows)
            block = self._store.put_file(path)
            self._keep(path, block)
            return ArchivedCorpus(block, units, window_count, token_count)

    def write_manifest(self, manifest: TokenisationManifest) -> ArtifactRef:
        message = self._manifests.assemble(manifest)
        return self._store.put(self._json.encode(message), Retention.DURABLE)

    def read_manifest(self, ref: ArtifactRef) -> TokenisationManifest:
        message = self._json.decode(self._store.get(ref))
        try:
            return self._manifests.restore(message)
        except ValueError as error:
            raise MalformedManifestError(
                f"manifest under {ref.key!r} breaks the Catalog's rules: {error}"
            ) from error

    def read_windows(
        self, archived: ArchivedCorpus, units: Collection[UnitKey]
    ) -> Sequence[TokenWindow]:
        try:
            block = self._blocks.open(archived.block)
        except MalformedBlockError as error:
            raise UnreadableCorpusBlockError(
                f"block {archived.block.key!r} is not one this archive reads: {error}"
            ) from error
        positions = {key: index for index, key in enumerate(archived.units)}
        return block.of_units({positions[key] for key in units if key in positions})

    @staticmethod
    def _write_block(
        path: Path, windows: Iterable[PlacedWindow]
    ) -> tuple[tuple[UnitKey, ...], int, int]:
        """Write the windows at ``path``; report the units in index order and the counts."""
        units: list[UnitKey] = []
        positions: dict[UnitKey, int] = {}
        window_count = token_count = 0
        with WindowBlockWriter(path, scratch=path.parent) as writer:
            for placed in windows:
                if placed.unit not in positions:
                    positions[placed.unit] = len(units)
                    units.append(placed.unit)
                try:
                    writer.add(
                        placed.window,
                        unit=positions[placed.unit],
                        start=placed.extent.start,
                        end=placed.extent.end,
                    )
                except UnstorableWindowError as error:
                    raise WindowNotArchivableError(
                        f"window {window_count} of unit {placed.unit} cannot be archived: {error}"
                    ) from error
                window_count += 1
                token_count += len(placed.window)
        return tuple(units), window_count, token_count

    def _keep(self, path: Path, block: ArtifactRef) -> None:
        cached = self._workspace / block.checksum.digest
        if not cached.exists():
            os.replace(path, cached)
