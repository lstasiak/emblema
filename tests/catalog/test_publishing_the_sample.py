"""Publishing run end to end on the miniature C-MAPSS sample, through the real reader and store."""

from collections.abc import Iterator
from itertools import chain
from pathlib import Path

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.entrypoints.cli.composition_root import CompositionRoot
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from tests.support.corpora import CORPUS, SAMPLE, SAMPLE_WINDOW, SUBSET, publish_command
from tests.support.settings import unreachable_store


def published(root: Path, registry: CorpusRepository) -> tuple[CompositionRoot, ArtifactRef]:
    process = CompositionRoot(
        unreachable_store(),
        corpus_root=SAMPLE,
        workspace=root / "blocks",
        subsets=(SUBSET,),
        corpora=registry,
        store=LocalDirectoryArtifactStore(root / "store"),
    )
    return process, process.services.publish_corpus(publish_command())


def tokenised() -> Iterator[TokenWindow]:
    """The windows the reader and the tokeniser produce, without any artifact in between."""
    reader = CmapssCorpusReader(SAMPLE, (SUBSET,))
    tokeniser = SlidingWindowTokeniser()
    units = list(reader.read_units())
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        CORPUS, reader.describe().channel_schema
    )
    scheme = tokeniser.fit(
        CORPUS,
        chain.from_iterable(reader.read_observations(unit.key) for unit in units),
        (),
        scheme,
    )
    for unit in units:
        for placed in tokeniser.tokenise(
            CORPUS, unit, reader.read_observations(unit.key), scheme, SAMPLE_WINDOW
        ):
            yield placed.window


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[CompositionRoot, ArtifactRef]:
    return published(tmp_path_factory.mktemp("published"), InMemoryCorpusRepository())


def test_the_published_corpus_holds_the_windows_the_tokeniser_cut(
    sample: tuple[CompositionRoot, ArtifactRef],
) -> None:
    process, ref = sample

    manifest = process.adapters.archive.read_manifest(ref)
    windows = process.adapters.archive.read_windows(manifest.archived, manifest.units)

    # Statistics are fitted on the training units alone here, so the values differ from a scheme
    # fitted on everything; what has to agree is which windows there are and how large they are.
    cut = list(tokenised())
    assert manifest.window_count == len(cut)
    assert [len(window) for window in windows] == [len(window) for window in cut]


def test_the_manifest_covers_every_unit_of_the_sample(
    sample: tuple[CompositionRoot, ArtifactRef],
) -> None:
    process, ref = sample

    manifest = process.adapters.archive.read_manifest(ref)

    assert len(manifest.units) == 2
    assert len(manifest.split.training) == 1
    assert len(manifest.split.validation) == 1
    assert manifest.token_count == sum(len(window) for window in tokenised())


def test_publishing_the_same_corpus_twice_gives_one_block_and_one_manifest(
    tmp_path: Path,
) -> None:
    # The identity of a published corpus is the checksum of its bytes, so the same data and the
    # same configuration have to produce the same artifact; with one registry across the two
    # runs, the same description of it too.
    registry = InMemoryCorpusRepository()
    first_process, first_ref = published(tmp_path / "first", registry)
    second_process, second_ref = published(tmp_path / "second", registry)

    first = first_process.adapters.archive.read_manifest(first_ref)
    second = second_process.adapters.archive.read_manifest(second_ref)
    assert first.block == second.block
    assert first_ref == second_ref


def test_a_registry_that_forgets_yields_a_second_manifest_for_the_same_block(
    tmp_path: Path,
) -> None:
    # Two registries stand for two processes with nothing persisted between them: the data is one
    # artifact, its description is minted twice.
    first_process, first_ref = published(tmp_path / "first", InMemoryCorpusRepository())
    second_process, second_ref = published(tmp_path / "second", InMemoryCorpusRepository())

    first = first_process.adapters.archive.read_manifest(first_ref)
    second = second_process.adapters.archive.read_manifest(second_ref)
    assert first_ref != second_ref
    assert first.block == second.block
    assert first.corpus_version != second.corpus_version
