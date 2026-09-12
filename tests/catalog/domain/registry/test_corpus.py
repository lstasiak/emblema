import pytest

from emblema.catalog.domain.channels.channel_schema import ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusVersionAlreadyExistsError,
    CorpusVersionFrozenError,
    CorpusVersionNotFoundError,
    InvalidCorpusError,
    SameDataAlreadyFrozenError,
)
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.catalog.domain.registry.corpus_version import CorpusVersion
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    OTHER_SCHEMA,
    SCHEMA,
    SOURCE,
    content,
    corpus_id,
    empty_corpus,
    instant,
    version_id,
)

REGULAR = SamplingRegime.REGULAR
IRREGULAR = SamplingRegime.IRREGULAR


@pytest.fixture
def corpus() -> Corpus:
    return empty_corpus()


@pytest.fixture
def with_frozen_v1(corpus: Corpus) -> Corpus:
    return (
        corpus.add_version(version_id(1), SCHEMA, REGULAR, LICENCE)
        .record_content(version_id(1), content())
        .freeze_version(version_id(1), AT)
    )


def test_versions_are_numbered_from_one_in_order(corpus: Corpus) -> None:
    grown = corpus.add_version(version_id(1), SCHEMA, REGULAR, LICENCE).add_version(
        version_id(2), OTHER_SCHEMA, IRREGULAR, LICENCE
    )

    assert [version.number for version in grown.versions] == [1, 2]
    assert grown.get_version(version_id(2)).sampling_regime is IRREGULAR


def test_operations_leave_the_receiving_corpus_unchanged(corpus: Corpus) -> None:
    corpus.add_version(version_id(1), SCHEMA, REGULAR, LICENCE)

    assert corpus.versions == ()


def test_unknown_version_is_reported(corpus: Corpus) -> None:
    with pytest.raises(CorpusVersionNotFoundError):
        corpus.get_version(version_id(9))
    with pytest.raises(CorpusVersionNotFoundError):
        corpus.record_content(version_id(9), content())
    with pytest.raises(CorpusVersionNotFoundError):
        corpus.freeze_version(version_id(9), AT)


def test_record_then_freeze_makes_the_version_frozen(with_frozen_v1: Corpus) -> None:
    version = with_frozen_v1.get_version(version_id(1))

    assert version.is_frozen
    assert version.content == content()
    assert version.frozen_at == AT
    assert with_frozen_v1.frozen_versions == (version,)


def test_frozen_version_rejects_a_content_change(with_frozen_v1: Corpus) -> None:
    with pytest.raises(CorpusVersionFrozenError):
        with_frozen_v1.record_content(version_id(1), content(b"modified"))


def test_changed_data_needs_a_new_version(with_frozen_v1: Corpus) -> None:
    grown = (
        with_frozen_v1.add_version(version_id(2), SCHEMA, REGULAR, LICENCE)
        .record_content(version_id(2), content(b"modified"))
        .freeze_version(version_id(2), instant(1))
    )

    assert [version.number for version in grown.frozen_versions] == [1, 2]


def test_unchanged_data_cannot_become_a_second_frozen_version(with_frozen_v1: Corpus) -> None:
    twin = with_frozen_v1.add_version(version_id(2), SCHEMA, REGULAR, LICENCE).record_content(
        version_id(2), content()
    )

    with pytest.raises(SameDataAlreadyFrozenError, match="version 1"):
        twin.freeze_version(version_id(2), instant(1))


@pytest.mark.parametrize(
    ("schema", "regime"),
    [(SCHEMA, IRREGULAR), (OTHER_SCHEMA, REGULAR)],
    ids=["other-regime", "other-schema"],
)
def test_same_checksum_under_another_schema_or_regime_is_a_distinct_version(
    with_frozen_v1: Corpus, schema: ChannelSchema, regime: SamplingRegime
) -> None:
    grown = (
        with_frozen_v1.add_version(version_id(2), schema, regime, LICENCE)
        .record_content(version_id(2), content())
        .freeze_version(version_id(2), instant(1))
    )

    assert len(grown.frozen_versions) == 2


def test_a_draft_does_not_count_as_a_duplicate(with_frozen_v1: Corpus) -> None:
    draft = with_frozen_v1.add_version(version_id(2), SCHEMA, REGULAR, LICENCE).record_content(
        version_id(2), content()
    )

    assert len(draft.frozen_versions) == 1


@pytest.mark.parametrize("name", ["", "  ", " C-MAPSS", "C-MAPSS "])
def test_corpus_name_must_be_non_blank_without_padding(name: str) -> None:
    with pytest.raises(InvalidCorpusError, match="name"):
        Corpus(corpus_id(), name, SOURCE)


def test_reusing_a_version_id_is_rejected(corpus: Corpus) -> None:
    grown = corpus.add_version(version_id(1), SCHEMA, REGULAR, LICENCE)

    with pytest.raises(CorpusVersionAlreadyExistsError):
        grown.add_version(version_id(1), OTHER_SCHEMA, IRREGULAR, LICENCE)


def test_reconstitution_rejects_gaps_in_numbering() -> None:
    versions = (
        CorpusVersion(version_id(1), 1, SCHEMA, REGULAR, LICENCE),
        CorpusVersion(version_id(2), 3, SCHEMA, REGULAR, LICENCE),
    )

    with pytest.raises(InvalidCorpusError, match=r"1\.\.n"):
        Corpus(corpus_id(), "C-MAPSS", SOURCE, versions)


def test_reconstitution_rejects_repeated_version_ids() -> None:
    versions = (
        CorpusVersion(version_id(1), 1, SCHEMA, REGULAR, LICENCE),
        CorpusVersion(version_id(1), 2, SCHEMA, REGULAR, LICENCE),
    )

    with pytest.raises(InvalidCorpusError, match="unique"):
        Corpus(corpus_id(), "C-MAPSS", SOURCE, versions)


def test_reconstitution_rejects_frozen_versions_describing_the_same_data() -> None:
    versions = (
        CorpusVersion(version_id(1), 1, SCHEMA, REGULAR, LICENCE, content(), AT),
        CorpusVersion(version_id(2), 2, SCHEMA, REGULAR, LICENCE, content(), instant(1)),
    )

    with pytest.raises(InvalidCorpusError, match="distinct"):
        Corpus(corpus_id(), "C-MAPSS", SOURCE, versions)
