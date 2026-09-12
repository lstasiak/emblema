"""Publishing run end to end on the miniature C-MAPSS sample, through the real reader and store.

Everything below the process is settled elsewhere: the archive against its contract, the use case
over adapters that hold nothing. What is left is whether the whole thing, on real bytes and a real
filesystem, publishes a corpus a run can read — and whether publishing it twice gives one artifact,
which is the promise the checksum of a published corpus makes.
"""

from collections.abc import Iterator
from itertools import chain
from pathlib import Path

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition import Services, build_services
from emblema.entrypoints.cli.publish_corpus import parse, publish
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "cmapss"
CORPUS = "cmapss"
# Short enough that the two sample engines yield several windows each.
WINDOW = WindowSpec(length=20.0, stride=7.0)


def settings() -> Settings:
    """Settings whose store is never reached: the composition below is handed its own archive."""
    return Settings(
        artifact_store=ArtifactStoreSettings(
            endpoint_url="http://127.0.0.1:3900",
            region="garage",
            bucket="emblema",
            key_prefix="test",
        )
    )


def published(root: Path) -> tuple[Services, ArtifactRef]:
    store = LocalDirectoryArtifactStore(root / "store")
    services = build_services(
        settings(),
        corpus_root=SAMPLE,
        workspace=root / "blocks",
        subsets=("FD001",),
        archive=BlockCorpusArchive(store, root / "blocks"),
    )
    arguments = parse(
        [
            *("--corpus", CORPUS, "--root", str(SAMPLE), "--subset", "FD001"),
            *("--window", str(WINDOW.length), "--stride", str(WINDOW.stride)),
            *("--validation-fraction", "0.5", "--seed", "1"),
        ]
    )
    return services, publish(services, arguments)


def tokenised() -> Iterator[TokenWindow]:
    """The windows the reader and the tokeniser produce, without any artifact in between."""
    reader = CmapssCorpusReader(SAMPLE, ("FD001",))
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
            CORPUS, unit, reader.read_observations(unit.key), scheme, WINDOW
        ):
            yield placed.window


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory) -> tuple[Services, ArtifactRef]:
    return published(tmp_path_factory.mktemp("published"))


def test_the_published_corpus_holds_the_windows_the_tokeniser_cut(
    sample: tuple[Services, ArtifactRef],
) -> None:
    services, ref = sample

    manifest = services.archive.read_manifest(ref)
    windows = services.archive.read_windows(manifest, manifest.units)

    # Statistics are fitted on the training units alone here, so the values differ from a scheme
    # fitted on everything; what has to agree is which windows there are and how large they are.
    cut = list(tokenised())
    assert manifest.window_count == len(cut)
    assert [len(window) for window in windows] == [len(window) for window in cut]


def test_the_manifest_covers_every_unit_of_the_sample(
    sample: tuple[Services, ArtifactRef],
) -> None:
    services, ref = sample

    manifest = services.archive.read_manifest(ref)

    assert len(manifest.units) == 2
    assert len(manifest.split.training) == 1
    assert len(manifest.split.validation) == 1
    assert manifest.token_count == sum(len(window) for window in tokenised())


def test_publishing_the_same_corpus_twice_gives_one_block(tmp_path: Path) -> None:
    # The identity of a published corpus is the checksum of its bytes, so the same data and the
    # same configuration have to produce the same artifact rather than one that merely matches.
    first_services, first_ref = published(tmp_path / "first")
    second_services, second_ref = published(tmp_path / "second")

    first = first_services.archive.read_manifest(first_ref)
    second = second_services.archive.read_manifest(second_ref)
    assert first.block == second.block
    assert first.window_count == second.window_count
