"""Contract of the CorpusRepository port, run against every adapter."""

from collections.abc import Callable

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.exceptions import CorpusNameTakenError, CorpusNotFoundError
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import (
    LICENCE,
    SCHEMA,
    SOURCE,
    corpus_id,
    empty_corpus,
    version_id,
)

ADAPTERS: dict[str, Callable[[], CorpusRepository]] = {"in_memory": InMemoryCorpusRepository}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def corpora(request: pytest.FixtureRequest) -> CorpusRepository:
    factory: Callable[[], CorpusRepository] = request.param
    return factory()


def test_unknown_corpus_is_reported(corpora: CorpusRepository) -> None:
    with pytest.raises(CorpusNotFoundError):
        corpora.get(corpus_id(9))


def test_saved_corpus_is_returned_as_stored(corpora: CorpusRepository) -> None:
    corpus = empty_corpus()

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
