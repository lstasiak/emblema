from emblema.catalog.domain.registry.corpus import Corpus
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    OTHER_SCHEMA,
    SCHEMA,
    description,
    empty_corpus,
    version_id,
)


def frozen_over(data: bytes = b"records") -> Corpus:
    told = description(data)
    return (
        empty_corpus()
        .add_version(version_id(), SCHEMA, told.sampling_regime, LICENCE)
        .record_content(version_id(), told.content)
        .freeze_version(version_id(), AT)
    )


def test_a_frozen_version_describes_the_data_it_was_frozen_over() -> None:
    corpus = frozen_over()

    found = corpus.frozen_version_describing(description())

    assert found is not None
    assert found.id == version_id()


def test_other_data_is_described_by_no_version() -> None:
    assert frozen_over().frozen_version_describing(description(b"other records")) is None


def test_the_same_bytes_under_another_schema_are_other_data() -> None:
    assert frozen_over().frozen_version_describing(description(schema=OTHER_SCHEMA)) is None


def test_a_draft_describes_the_data_but_is_never_the_version_found() -> None:
    # Content recorded, nothing frozen: the draft may still change, so a publication cannot
    # rest on it, however exactly its content matches.
    told = description()
    corpus = (
        empty_corpus()
        .add_version(version_id(), SCHEMA, told.sampling_regime, LICENCE)
        .record_content(version_id(), told.content)
    )

    assert corpus.versions[0].describes(told)
    assert corpus.frozen_version_describing(told) is None


def test_a_version_without_content_describes_nothing() -> None:
    told = description()
    corpus = empty_corpus().add_version(version_id(), SCHEMA, told.sampling_regime, LICENCE)

    assert not corpus.versions[0].describes(told)
