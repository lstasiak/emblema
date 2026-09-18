"""What the block adapter does with a workspace the contract corpus cannot express."""

from pathlib import Path

from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from tests.support.published import TRAINING_UNIT, publish

TRAINING = UnitKey(TRAINING_UNIT)


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


def test_a_block_already_in_the_workspace_is_not_fetched_again(tmp_path: Path) -> None:
    store = CountingStore(InMemoryArtifactStore())
    published = publish(store, tmp_path)
    windows = BlockCorpusWindows(PublishedCorpusBlocks(store, tmp_path / "workspace"))

    first = windows.windows_of(published.manifest, {TRAINING})
    second = windows.windows_of(published.manifest, {TRAINING})

    assert second == first
    assert store.fetches == 1
