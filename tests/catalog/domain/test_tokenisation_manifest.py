import pytest

from emblema.catalog.domain.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import InvalidTokenisationManifestError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.catalog.domain.support import CORPUS, OTHER_SCHEMA, SCHEMA, version_id

BLOCK = ArtifactRef("durable/sha256/" + "0" * 64, Checksum.of_bytes(b"block"))
FIRST, SECOND, THIRD = UnitKey("u1"), UnitKey("u2"), UnitKey("u3")
ARCHIVED = ArchivedCorpus(BLOCK, (FIRST, SECOND), 3, 8)
SCHEME = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(CORPUS, SCHEMA)
SPLIT = UnitSplit(training=frozenset({FIRST}), validation=frozenset({SECOND}))


def manifest(
    *,
    corpus: str = CORPUS,
    archived: ArchivedCorpus = ARCHIVED,
    scheme: TokenisationScheme = SCHEME,
    split: UnitSplit = SPLIT,
    empty_units: tuple[UnitKey, ...] = (),
) -> TokenisationManifest:
    return TokenisationManifest(
        corpus=corpus,
        corpus_version=version_id(),
        corpus_checksum=Checksum.of_bytes(b"raw"),
        archived=archived,
        window=WindowSpec(10.0, 5.0),
        scheme=scheme,
        split=split,
        split_seed=1,
        empty_units=empty_units,
    )


def test_the_manifest_repeats_what_the_archive_settled() -> None:
    described = manifest()

    assert described.block == BLOCK
    assert described.units == (FIRST, SECOND)
    assert (described.window_count, described.token_count) == (3, 8)


@pytest.mark.parametrize("name", ["", " cmapss", "cmapss "])
def test_the_corpus_name_is_strict(name: str) -> None:
    with pytest.raises(InvalidTokenisationManifestError, match="non-blank"):
        manifest(corpus=name)


def test_the_scheme_must_register_the_corpus() -> None:
    foreign = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        "other", OTHER_SCHEMA
    )

    with pytest.raises(InvalidTokenisationManifestError, match="registers no channel"):
        manifest(scheme=foreign)


def test_a_unit_cannot_be_both_indexed_by_the_block_and_empty() -> None:
    with pytest.raises(InvalidTokenisationManifestError, match="both indexed"):
        manifest(empty_units=(FIRST,))


@pytest.mark.parametrize(
    ("split", "empty_units"),
    [
        (UnitSplit(training=frozenset({FIRST}), validation=frozenset({THIRD})), ()),
        (SPLIT, (THIRD,)),
    ],
    ids=["a unit the corpus does not have", "an empty unit the split leaves out"],
)
def test_the_split_must_name_exactly_the_units_of_the_corpus(
    split: UnitSplit, empty_units: tuple[UnitKey, ...]
) -> None:
    with pytest.raises(InvalidTokenisationManifestError, match="same units"):
        manifest(split=split, empty_units=empty_units)


def test_an_empty_unit_sits_on_a_side_of_the_split_like_any_other() -> None:
    split = UnitSplit(training=frozenset({FIRST, THIRD}), validation=frozenset({SECOND}))

    described = manifest(split=split, empty_units=(THIRD,))

    assert described.empty_units == (THIRD,)
    assert THIRD not in described.units
