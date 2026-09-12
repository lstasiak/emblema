"""Contract of the CorpusRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``, like the S3 store. Both have to keep the same promises,
down to the error a taken name raises.
"""

import pytest
from sqlalchemy import Engine

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.persistence.corpus_repository import SqlAlchemyCorpusRepository
from emblema.catalog.domain.exceptions import CorpusNameTakenError, CorpusNotFoundError
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.catalog.domain.registry.corpus_source import CorpusSource
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import (
    AT,
    LICENCE,
    SCHEMA,
    SOURCE,
    corpus_id,
    description,
    empty_corpus,
    version_id,
)
from tests.support.database import clear_catalog, migrated_engine

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def corpora(request: pytest.FixtureRequest) -> CorpusRepository:
    if request.param == "in_memory":
        return InMemoryCorpusRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_catalog(engine)
    return SqlAlchemyCorpusRepository(engine)


def test_unknown_corpus_is_reported(corpora: CorpusRepository) -> None:
    with pytest.raises(CorpusNotFoundError):
        corpora.get(corpus_id(9))


def test_saved_corpus_is_returned_as_stored(corpora: CorpusRepository) -> None:
    corpus = empty_corpus()

    corpora.save(corpus)

    assert corpora.get(corpus.id) == corpus


def test_a_corpus_with_versions_comes_back_whole(corpora: CorpusRepository) -> None:
    # A draft and a frozen version side by side: every value object of a version, with and
    # without content, has to survive the store and come back equal.
    told = description()
    corpus = (
        empty_corpus()
        .add_version(version_id(1), SCHEMA, told.sampling_regime, LICENCE)
        .record_content(version_id(1), told.content)
        .freeze_version(version_id(1), AT)
        .add_version(version_id(2), SCHEMA, SamplingRegime.IRREGULAR, LICENCE)
    )

    corpora.save(corpus)

    assert corpora.get(corpus.id) == corpus


def test_saving_again_replaces_the_stored_state(corpora: CorpusRepository) -> None:
    corpus = empty_corpus()
    corpora.save(corpus)
    grown = corpus.add_version(version_id(), SCHEMA, SamplingRegime.REGULAR, LICENCE)

    corpora.save(grown)

    assert corpora.get(corpus.id) == grown


def test_a_name_belongs_to_one_corpus(corpora: CorpusRepository) -> None:
    corpora.save(empty_corpus())

    with pytest.raises(CorpusNameTakenError):
        corpora.save(Corpus(corpus_id(2), "C-MAPSS", SOURCE))


def test_corpora_of_distinct_names_coexist(corpora: CorpusRepository) -> None:
    first, second = empty_corpus(), Corpus(corpus_id(2), "SKAB", SOURCE)

    corpora.save(first)
    corpora.save(second)

    assert (corpora.get(first.id), corpora.get(second.id)) == (first, second)


def test_a_name_nobody_registered_finds_nothing(corpora: CorpusRepository) -> None:
    assert corpora.find_by_name("nobody") is None


def test_a_corpus_is_found_by_its_name(corpora: CorpusRepository) -> None:
    corpus = empty_corpus()
    corpora.save(corpus)

    assert corpora.find_by_name(corpus.name) == corpus


def test_a_corpus_is_found_by_its_current_name_only(corpora: CorpusRepository) -> None:
    corpus = empty_corpus()
    corpora.save(corpus)
    renamed = Corpus(corpus.id, "C-MAPSS (turbofan)", CorpusSource(SOURCE.name, SOURCE.uri))

    corpora.save(renamed)

    assert corpora.find_by_name(renamed.name) == renamed
    assert corpora.find_by_name(corpus.name) is None
