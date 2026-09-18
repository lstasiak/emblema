from pathlib import Path

from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore


class BlockWorkspace:
    """A directory blocks are fetched to once and mapped from, each under its digest.

    A block travels whole and is read many times, so it is kept where it lands: a machine that
    published a corpus reads it back without fetching it again, and any machine fetches it once.
    A block found in the workspace is hashed before it is mapped — the directory is shared with
    other processes and outlives every one of them, and a run over bytes nobody verified would
    sign as a run over the corpus. It is hashed once per workspace, not once per opening: a
    process that reads one block through several readers verifies it the first time and trusts
    its own verification afterwards.
    """

    def __init__(self, store: ArtifactStore, root: Path) -> None:
        self._store = store
        self._root = root
        self._verified: dict[ArtifactRef, Path] = {}

    def open(self, block: ArtifactRef) -> WindowBlock:
        """The block mapped from the workspace, fetched first where it is absent or not intact.

        Raises:
            MalformedBlockError: If the artifact is not a block this reads.
            ArtifactNotFoundError: If the block is not in the store.
            ArtifactIntegrityError: If the stored block does not hash to its checksum.
        """
        return WindowBlock(self.fetched(block))

    def fetched(self, block: ArtifactRef) -> Path:
        """Where the block's bytes lie in the workspace, verified against its checksum once."""
        if block in self._verified:
            return self._verified[block]
        path = self._root / block.checksum.digest
        if not (
            path.is_file()
            and Checksum.of_chunks(chunks_of(path), block.checksum.algorithm) == block.checksum
        ):
            self._root.mkdir(parents=True, exist_ok=True)
            self._store.get_file(block, path)
        self._verified[block] = path
        return path
