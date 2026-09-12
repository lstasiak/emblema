"""Contract of the CorpusArchive port, run against every store the one adapter writes into.

A corpus is archived by one adapter — windows in a block, the description beside it — and what
varies beneath is where the bytes land. So the contract is parametrised by the store: the same
corpus written into memory and onto a disk has to come back the same corpus, under the same
reference, because that reference is the identity of a published corpus.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.corpus_unit import TimeExtent
from emblema.catalog.domain.exceptions import MalformedManifestError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.catalog.ports.corpus_archive import CorpusArchive
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.catalog.domain.support import SCHEMA, version_id
from tests.shared.adapters.windows.test_window_block import TIMED, WITH_STATIC

STORES: dict[str, Callable[[Path], ArtifactStore]] = {
    "in_memory": lambda _: InMemoryArtifactStore(),
    "local_directory": lambda root: LocalDirectoryArtifactStore(root / "artifacts"),
}
FIRST = UnitKey("u1")
SECOND = UnitKey("u2")
PLACED = (
    PlacedWindow(FIRST, TimeExtent(0.0, 10.0), TIMED),
    PlacedWindow(FIRST, TimeExtent(5.0, 15.0), WITH_STATIC),
    PlacedWindow(SECOND, TimeExtent(0.0, 10.0), TIMED),
)


class Harness(NamedTuple):
    """An archive under test plus the store it writes into, for reaching past the port."""

    archive: CorpusArchive
    store: ArtifactStore


@pytest.fixture(params=list(STORES.values()), ids=list(STORES))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Harness:
    build: Callable[[Path], ArtifactStore] = request.param
    store = build(tmp_path)
    return Harness(BlockCorpusArchive(store, tmp_path / "workspace"), store)


@pytest.fixture
def archive(harness: Harness) -> CorpusArchive:
    return harness.archive


def manifest_of(
    block: ArtifactRef, units: Sequence[UnitKey], windows: int = 3, tokens: int = 8
) -> TokenisationManifest:
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with("corpus", SCHEMA)
    return TokenisationManifest(
        corpus="corpus",
        corpus_version=version_id(),
        corpus_checksum=Checksum.of_bytes(b"raw"),
        block=block,
        window=WindowSpec(10.0, 5.0),
        scheme=scheme,
        units=tuple(units),
        split=UnitSplit(training=frozenset({FIRST}), validation=frozenset({SECOND})),
        split_seed=1,
        window_count=windows,
        token_count=tokens,
    )


def archived(archive: CorpusArchive) -> TokenisationManifest:
    written = archive.write_windows(PLACED)
    return manifest_of(written.block, written.units, written.window_count, written.token_count)


def test_the_windows_of_a_corpus_come_back_as_they_went_in(archive: CorpusArchive) -> None:
    manifest = archived(archive)

    read = archive.read_windows(manifest, manifest.units)

    assert list(read) == [placed.window for placed in PLACED]


def test_only_the_windows_of_the_units_asked_for_come_back(archive: CorpusArchive) -> None:
    manifest = archived(archive)

    assert list(archive.read_windows(manifest, [FIRST])) == [TIMED, WITH_STATIC]
    assert list(archive.read_windows(manifest, [SECOND])) == [TIMED]
    assert list(archive.read_windows(manifest, [])) == []


def test_the_units_come_back_in_the_order_their_windows_arrived(archive: CorpusArchive) -> None:
    written = archive.write_windows(PLACED)

    assert written.units == (FIRST, SECOND)
    assert written.window_count == 3
    assert written.token_count == len(TIMED) * 2 + len(WITH_STATIC)


def test_the_same_corpus_archived_twice_is_one_artifact(archive: CorpusArchive) -> None:
    # The reference is the identity of a published corpus: two runs of one configuration have to
    # produce one artifact, not two that happen to hold the same numbers.
    first = archive.write_windows(PLACED)
    second = archive.write_windows(PLACED)

    assert first.block == second.block


def test_a_corpus_that_yields_no_window_is_archived_all_the_same(archive: CorpusArchive) -> None:
    written = archive.write_windows([])

    assert written.units == ()
    assert written.window_count == 0
    assert written.token_count == 0


def test_a_manifest_comes_back_as_it_was_written(archive: CorpusArchive) -> None:
    manifest = archived(archive)

    ref = archive.write_manifest(manifest)

    assert archive.read_manifest(ref) == manifest


def test_the_same_manifest_written_twice_is_one_artifact(archive: CorpusArchive) -> None:
    manifest = archived(archive)

    assert archive.write_manifest(manifest) == archive.write_manifest(manifest)


def test_reading_a_manifest_that_was_never_written_fails(archive: CorpusArchive) -> None:
    with pytest.raises(ArtifactNotFoundError):
        archive.read_manifest(content_address(b"never stored", Retention.DURABLE))


def test_reading_windows_whose_block_is_gone_fails(archive: CorpusArchive) -> None:
    manifest = archived(archive)
    elsewhere = manifest_of(
        ArtifactRef("durable/sha256/" + "0" * 64, Checksum.of_bytes(b"nothing")), manifest.units
    )

    with pytest.raises(ArtifactNotFoundError):
        archive.read_windows(elsewhere, elsewhere.units)


def test_bytes_that_are_not_a_manifest_are_refused(harness: Harness) -> None:
    ref = harness.store.put(b"not a manifest")

    with pytest.raises(MalformedManifestError):
        harness.archive.read_manifest(ref)
