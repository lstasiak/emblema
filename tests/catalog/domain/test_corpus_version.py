import pytest

from emblema.catalog.domain.corpus_version import CorpusVersion
from emblema.catalog.domain.exceptions import (
    CorpusVersionFrozenError,
    CorpusVersionNotValidatedError,
    InvalidCorpusVersionError,
)
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    OTHER_SCHEMA,
    SCHEMA,
    content,
    instant,
    version_id,
)


@pytest.fixture
def draft() -> CorpusVersion:
    return CorpusVersion(version_id(), 1, SCHEMA, SamplingRegime.REGULAR, LICENCE)


def test_new_version_is_a_draft_without_content(draft: CorpusVersion) -> None:
    assert not draft.is_frozen
    assert draft.content is None
    assert draft.frozen_at is None


def test_with_content_returns_a_new_version_and_keeps_the_original(draft: CorpusVersion) -> None:
    validated = draft.with_content(content())

    assert validated.content == content()
    assert draft.content is None


def test_content_can_be_re_recorded_while_draft(draft: CorpusVersion) -> None:
    revalidated = draft.with_content(content(b"first")).with_content(content(b"second"))

    assert revalidated.content == content(b"second")


def test_freeze_requires_validated_content(draft: CorpusVersion) -> None:
    with pytest.raises(CorpusVersionNotValidatedError):
        draft.freeze(AT)


def test_freeze_records_the_instant(draft: CorpusVersion) -> None:
    frozen = draft.with_content(content()).freeze(AT)

    assert frozen.is_frozen
    assert frozen.frozen_at == AT


def test_frozen_version_rejects_new_content(draft: CorpusVersion) -> None:
    frozen = draft.with_content(content()).freeze(AT)

    with pytest.raises(CorpusVersionFrozenError, match="new version"):
        frozen.with_content(content(b"changed"))


def test_frozen_version_rejects_a_second_freeze(draft: CorpusVersion) -> None:
    frozen = draft.with_content(content()).freeze(AT)

    with pytest.raises(CorpusVersionFrozenError):
        frozen.freeze(instant(1))


def test_versions_with_equal_checksum_schema_and_regime_describe_the_same_data(
    draft: CorpusVersion,
) -> None:
    first = draft.with_content(content())
    second = CorpusVersion(version_id(2), 2, SCHEMA, SamplingRegime.REGULAR, LICENCE, content())

    assert first.describes_same_data_as(second)
    assert second.describes_same_data_as(first)


@pytest.mark.parametrize(
    "other",
    [
        CorpusVersion(version_id(2), 2, SCHEMA, SamplingRegime.IRREGULAR, LICENCE, content()),
        CorpusVersion(version_id(2), 2, OTHER_SCHEMA, SamplingRegime.REGULAR, LICENCE, content()),
        CorpusVersion(version_id(2), 2, SCHEMA, SamplingRegime.REGULAR, LICENCE, content(b"other")),
    ],
    ids=["other-regime", "other-schema", "other-checksum"],
)
def test_any_difference_in_regime_schema_or_checksum_means_different_data(
    draft: CorpusVersion, other: CorpusVersion
) -> None:
    assert not draft.with_content(content()).describes_same_data_as(other)


def test_a_version_without_content_describes_no_data(draft: CorpusVersion) -> None:
    validated = draft.with_content(content())

    assert not draft.describes_same_data_as(validated)
    assert not validated.describes_same_data_as(draft)


def test_frozen_version_without_content_cannot_exist() -> None:
    with pytest.raises(InvalidCorpusVersionError, match="content"):
        CorpusVersion(version_id(), 1, SCHEMA, SamplingRegime.REGULAR, LICENCE, frozen_at=AT)


@pytest.mark.parametrize("number", [0, -1])
def test_version_number_must_be_positive(number: int) -> None:
    with pytest.raises(InvalidCorpusVersionError, match="positive"):
        CorpusVersion(version_id(), number, SCHEMA, SamplingRegime.REGULAR, LICENCE)
