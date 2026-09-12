import tempfile
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from emblema.catalog.adapters.archive.manifest_json import decode_manifest, encode_manifest
from emblema.catalog.domain.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore, Retention


class BlockCorpusArchive:
    """A corpus archived as a block of windows beside a manifest, both in an artifact store.

    The block never passes through memory: it is written to a working directory a window at a
    time, handed to the store as a file, and fetched back as a file so that a reader can map it.
    That working directory is where a fetched corpus stays, so a second run over the same corpus
    reads the disk rather than the network — the file is content-addressed, so a name that is
    already there is already the right bytes.
    """

    def __init__(self, store: ArtifactStore, workspace: Path) -> None:
        """Archive into ``store``, using ``workspace`` for blocks on their way in or out."""
        self._store = store
        self._workspace = workspace

    def write_windows(self, windows: Iterable[PlacedWindow]) -> ArchivedCorpus:
        self._workspace.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self._workspace) as scratch:
            path = Path(scratch) / "corpus.block"
            units: list[UnitKey] = []
            positions: dict[UnitKey, int] = {}
            written = tokens = 0
            with WindowBlockWriter(path) as writer:
                for placed in windows:
                    if placed.unit not in positions:
                        positions[placed.unit] = len(units)
                        units.append(placed.unit)
                    writer.add(
                        positions[placed.unit],
                        placed.extent.start,
                        placed.extent.end,
                        placed.window,
                    )
                    written += 1
                    tokens += len(placed.window)
            return ArchivedCorpus(
                block=self._store.put_file(path),
                units=tuple(units),
                window_count=written,
                token_count=tokens,
            )

    def write_manifest(self, manifest: TokenisationManifest) -> ArtifactRef:
        return self._store.put(encode_manifest(manifest), Retention.DURABLE)

    def read_manifest(self, ref: ArtifactRef) -> TokenisationManifest:
        return decode_manifest(self._store.get(ref))

    def read_windows(
        self, manifest: TokenisationManifest, units: Collection[UnitKey]
    ) -> Sequence[TokenWindow]:
        block = WindowBlock(self._fetched(manifest.block))
        positions = {key: index for index, key in enumerate(manifest.units)}
        return block.of_units({positions[key] for key in units if key in positions})

    def _fetched(self, block: ArtifactRef) -> Path:
        path = self._workspace / block.checksum.digest
        if not path.is_file():
            self._store.get_file(block, path)
        return path
