"""Every downloaded corpus travels the same road, on its miniature sample.

The point of a new reader is that nothing behind it has to change: the same command line, the same
use case, the same tokeniser and the same archive turn a corpus of 8, 21 or 38 channels into an
artifact, and 17 channels of two satellites into one as well. This module asserts that once per
corpus, shallowly. What a published corpus has to hold is asserted deeply for one of them in
``test_publishing_the_sample``.
"""

from importlib.util import find_spec
from pathlib import Path

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.split_policy import NamedSplit, SplitPolicy
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from tests.support.corpora import DRAWN, SAMPLE_WINDOW, publish_command, sample
from tests.support.settings import unreachable_store

# Corpus, the subsets its sample holds, how many units with windows and channels that makes, the
# window in the corpus's own unit of time, and the split. The satellite sample keeps one row in
# two days, so its windows are ten days long where the others are twenty cycles, seconds or
# minutes. A clinical stay is windowed whole, and the stay that holds descriptors alone yields no
# window; its stays are split by name, because a channel measured only on the held-out side has no
# statistics to be normalised by, and four stays are too few for a seeded draw to avoid that.
CORPORA = [
    ("cmapss", ("FD001",), 2, 21, SAMPLE_WINDOW, DRAWN),
    ("skab", ("anomaly-free", "other", "valve1"), 3, 8, SAMPLE_WINDOW, DRAWN),
    ("smd", ("1", "2"), 2, 38, SAMPLE_WINDOW, DRAWN),
    pytest.param(
        "esa_ad",
        ("ESA-Mission1", "ESA-Mission2"),
        22,
        17,
        WindowSpec(length=240.0, stride=120.0),
        DRAWN,
        marks=pytest.mark.skipif(
            find_spec("pandas") is None, reason="the corpora extra is not installed"
        ),
    ),
    (
        "physionet2012",
        ("set-a", "set-b"),
        3,
        44,
        WindowSpec(length=48.0, stride=48.0),
        NamedSplit.of([UnitKey("set-a/132539"), UnitKey("set-a/140501")]),
    ),
]


@pytest.mark.parametrize(
    ("corpus", "subsets", "units", "channels", "window", "split"),
    CORPORA,
    ids=["cmapss", "skab", "smd", "esa_ad", "physionet2012"],
)
def test_a_sample_of_each_corpus_publishes_through_one_process(
    tmp_path: Path,
    corpus: str,
    subsets: tuple[str, ...],
    units: int,
    channels: int,
    window: WindowSpec,
    split: SplitPolicy,
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

    ref = process.services.publish_corpus(
        publish_command(corpus=corpus, window=window, split=split)
    )

    manifest = process.adapters.archive.read_manifest(ref)
    assert len(manifest.units) == units
    assert len(manifest.scheme.vocabulary.entries_of(corpus)) == channels
    assert manifest.archived.window_count > 0
    assert manifest.archived.token_count == sum(
        len(window)
        for window in process.adapters.archive.read_windows(manifest.archived, manifest.units)
    )
