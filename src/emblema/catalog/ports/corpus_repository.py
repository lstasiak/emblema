from typing import Protocol

from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.identifiers import CorpusId


class CorpusRepository(Protocol):
    """Persistence of the ``Corpus`` aggregate: whole corpora in, whole corpora out.

    The corpus is immutable, so ``save`` stores the state the caller holds and replaces whatever
    was stored under the same identity; one call is one transaction. Corpus names are unique
    across the repository, a rule no single aggregate can see, which is why it is enforced here.
    """

    def get(self, corpus_id: CorpusId) -> Corpus:
        """The stored state of one corpus.

        Raises:
            CorpusNotFoundError: If no corpus has that identifier.
        """
        ...

    def save(self, corpus: Corpus) -> None:
        """Store this state of the corpus, replacing the previous one.

        Raises:
            CorpusNameTakenError: If another corpus already carries the same name.
        """
        ...
