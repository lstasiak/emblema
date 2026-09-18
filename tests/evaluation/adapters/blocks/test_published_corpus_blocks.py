"""What the shared reader of published corpora refuses, in this context's words."""

from pathlib import Path

import pytest

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.support.published import manifest_of


def test_an_artifact_that_is_not_a_manifest_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()

    with pytest.raises(UnreadableTaskCorpusError, match="not a published manifest"):
        PublishedCorpusBlocks(store, tmp_path / "workspace").manifest_of(store.put(b"{}"))


def test_a_manifest_pointing_at_something_that_is_not_a_block_is_refused(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    published = manifest_of(store.put(b"not a block"))

    with pytest.raises(UnreadableTaskCorpusError, match="is not one this reads"):
        PublishedCorpusBlocks(store, tmp_path / "workspace").block_of(published)
