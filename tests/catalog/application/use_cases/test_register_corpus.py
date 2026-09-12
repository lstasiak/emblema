from uuid import UUID

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.application.use_cases.register_corpus import (
    RegisterCorpus,
    RegisterCorpusCommand,
)
from emblema.catalog.domain.exceptions import CorpusNameTakenError, InvalidCorpusError
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.registry.corpus import Corpus
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.catalog.domain.support import SOURCE


@pytest.fixture
def corpora() -> InMemoryCorpusRepository:
    return InMemoryCorpusRepository()


@pytest.fixture
def register(corpora: InMemoryCorpusRepository) -> RegisterCorpus:
    return RegisterCorpus(corpora, SequentialIdGenerator())


def command(name: str = "C-MAPSS") -> RegisterCorpusCommand:
    return RegisterCorpusCommand(name=name, source=SOURCE)


def test_stores_a_corpus_without_versions(
    register: RegisterCorpus, corpora: InMemoryCorpusRepository
) -> None:
    corpus_id = register(command())

    assert corpora.get(corpus_id) == Corpus(CorpusId(UUID(int=1)), "C-MAPSS", SOURCE)


def test_a_second_corpus_of_the_same_name_is_rejected(register: RegisterCorpus) -> None:
    register(command())

    with pytest.raises(CorpusNameTakenError):
        register(command())


def test_a_padded_name_is_rejected(register: RegisterCorpus) -> None:
    with pytest.raises(InvalidCorpusError):
        register(command(" C-MAPSS"))
