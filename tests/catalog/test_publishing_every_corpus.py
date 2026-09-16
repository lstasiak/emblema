"""Every downloaded corpus travels the same road, on its miniature sample.

The point of a new reader is that nothing behind it has to change: the same command line, the same
use case, the same tokeniser and the same archive turn a corpus of 8, 21 or 38 channels into an
artifact. This module asserts that once per corpus, shallowly. What a published corpus has to hold
is asserted deeply for one of them in ``test_publishing_the_sample``.
"""

from pathlib import Path

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from tests.support.corpora import SAMPLE_WINDOW, publish_command, sample
from tests.support.settings import unreachable_store

# Corpus, the subsets its sample holds, and how many units and channels that makes.
CORPORA = [
    ("cmapss", ("FD001",), 2, 21),
    ("skab", ("anomaly-free", "other", "valve1"), 3, 8),
    ("smd", ("1", "2"), 2, 38),
]


@pytest.mark.parametrize(
    ("corpus", "subsets", "units", "channels"), CORPORA, ids=[row[0] for row in CORPORA]
)
def test_a_sample_of_each_corpus_publishes_through_one_process(
    tmp_path: Path, corpus: str, subsets: tuple[str, ...], units: int, channels: int
) -> None:
    process = CompositionRoot(
        unreachable_store(),
        corpus=corpus,
        corpus_root=sample(corpus),
        workspace=tmp_path / "blocks",
        subsets=subsets,
        corpora=InMemoryCorpusRepository(),
        store=LocalDirectoryArtifactStore(tmp_path / "store"),
    )

    ref = process.services.publish_corpus(publish_command(corpus=corpus, window=SAMPLE_WINDOW))

    manifest = process.adapters.archive.read_manifest(ref)
    assert len(manifest.units) == units
    assert len(manifest.scheme.vocabulary.entries_of(corpus)) == channels
    assert manifest.archived.window_count > 0
    assert manifest.archived.token_count == sum(
        len(window)
        for window in process.adapters.archive.read_windows(manifest.archived, manifest.units)
    )
