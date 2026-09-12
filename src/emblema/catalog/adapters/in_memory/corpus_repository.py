from emblema.catalog.domain.exceptions import CorpusNameTakenError, CorpusNotFoundError
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.registry.corpus import Corpus


class InMemoryCorpusRepository:
    """Repository over a dictionary: the fake of the port for application tests."""

    def __init__(self) -> None:
        self._corpora: dict[CorpusId, Corpus] = {}

    def get(self, corpus_id: CorpusId) -> Corpus:
        try:
            return self._corpora[corpus_id]
        except KeyError:
            raise CorpusNotFoundError(f"no corpus {corpus_id}") from None

    def find_by_name(self, name: str) -> Corpus | None:
        return next((corpus for corpus in self._corpora.values() if corpus.name == name), None)

    def save(self, corpus: Corpus) -> None:
        for other in self._corpora.values():
            if other.id != corpus.id and other.name == corpus.name:
                raise CorpusNameTakenError(f"corpus name {corpus.name!r} is taken by {other.id}")
        self._corpora[corpus.id] = corpus
