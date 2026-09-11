from dataclasses import dataclass

from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.ports.corpus_repository import CorpusRepository
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class RegisterCorpusCommand:
    name: str
    source: CorpusSource


class RegisterCorpus:
    """Admits a corpus without any version yet; versions are registered one reading at a time.

    Creating the corpus and reading its data are kept apart on purpose: a typo in a name must
    fail as an unknown corpus, never silently create a second one.
    """

    def __init__(self, corpora: CorpusRepository, ids: IdGenerator) -> None:
        self._corpora = corpora
        self._ids = ids

    def __call__(self, command: RegisterCorpusCommand) -> CorpusId:
        """Store the new corpus and return its identity.

        Raises:
            InvalidCorpusError: If the name is blank or padded.
            CorpusNameTakenError: If a corpus of that name already exists.
        """
        corpus = Corpus(self._ids.generate(CorpusId), command.name, command.source)
        self._corpora.save(corpus)
        return corpus.id
