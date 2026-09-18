from pathlib import Path

import pytest

from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.windows.block_workspace import BlockWorkspace
from emblema.shared.adapters.windows.exceptions import MalformedBlockError
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from tests.shared.adapters.windows.support import TIMED, WITH_STATIC


class CountingStore:
    """A store that says how often it was asked to fetch a file."""

    def __init__(self, inner: ArtifactStore) -> None:
        self._inner = inner
        self.fetches = 0

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self._inner.put(content, retention)

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        return self._inner.put_file(source, retention)

    def get(self, ref: ArtifactRef) -> bytes:
        return self._inner.get(ref)

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        self.fetches += 1
        self._inner.get_file(ref, destination)

    def exists(self, ref: ArtifactRef) -> bool:
        return self._inner.exists(ref)


@pytest.fixture
def store() -> CountingStore:
    return CountingStore(InMemoryArtifactStore())


def stored_block(store: ArtifactStore, scratch: Path) -> ArtifactRef:
    path = scratch / "corpus.block"
    with WindowBlockWriter(path, scratch=scratch) as writer:
        writer.add(TIMED, unit=0, start=0.0, end=10.0)
        writer.add(WITH_STATIC, unit=1, start=5.0, end=15.0)
    return store.put_file(path)


def test_a_block_opens_from_the_store_with_its_windows(
    store: CountingStore, tmp_path: Path
) -> None:
    block = stored_block(store, tmp_path / "scratch")

    opened = BlockWorkspace(store, tmp_path / "workspace").open(block)

    assert list(opened) == [TIMED, WITH_STATIC]


def test_a_block_already_in_the_workspace_is_not_fetched_again(
    store: CountingStore, tmp_path: Path
) -> None:
    block = stored_block(store, tmp_path / "scratch")
    workspace = BlockWorkspace(store, tmp_path / "workspace")

    workspace.open(block)
    workspace.open(block)

    assert store.fetches == 1


def test_a_file_in_the_workspace_that_does_not_hash_to_the_block_is_replaced(
    store: CountingStore, tmp_path: Path
) -> None:
    block = stored_block(store, tmp_path / "scratch")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / block.checksum.digest).write_bytes(b"not the block")

    opened = BlockWorkspace(store, root).open(block)

    assert store.fetches == 1
    assert list(opened) == [TIMED, WITH_STATIC]


def test_a_block_verified_once_is_trusted_by_the_same_workspace_afterwards(
    store: CountingStore, tmp_path: Path
) -> None:
    block = stored_block(store, tmp_path / "scratch")
    workspace = BlockWorkspace(store, tmp_path / "workspace")
    path = workspace.fetched(block)
    path.write_bytes(b"changed after the workspace verified it")

    assert workspace.fetched(block) == path
    assert store.fetches == 1


def test_an_artifact_that_is_not_a_block_is_refused(store: CountingStore, tmp_path: Path) -> None:
    not_a_block = store.put(b"not a block")

    with pytest.raises(MalformedBlockError):
        BlockWorkspace(store, tmp_path / "workspace").open(not_a_block)
