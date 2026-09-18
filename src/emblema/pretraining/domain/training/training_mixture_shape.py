from dataclasses import dataclass
from itertools import pairwise

from emblema.pretraining.domain.exceptions import InvalidTrainingMixtureError
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape


@dataclass(frozen=True, kw_only=True)
class TrainingMixtureShape:
    """What a run's data is, without the windows: the shape of each corpus, in the order read.

    A run is identified by its configuration and the corpora it read, and this is the corpora as
    a result made elsewhere states them and a registry holds them: enough to tell two runs' data
    apart, and to hold a result to an order, without a window of either being here.

    Invariants: at least one corpus; no corpus twice; the vocabularies do not shrink along the
    order, since each corpus's vocabulary continues the one before it.

    Attributes:
        corpora: The shape of each corpus, in the order the mixture reads them.
    """

    corpora: tuple[TrainingCorpusShape, ...]

    def __post_init__(self) -> None:
        if not self.corpora:
            raise InvalidTrainingMixtureError("a mixture holds at least one corpus")
        names = [corpus.name for corpus in self.corpora]
        if len(set(names)) != len(names):
            raise InvalidTrainingMixtureError(f"a corpus is mixed in twice: {names}")
        sizes = [corpus.vocabulary_size for corpus in self.corpora]
        if any(later < earlier for earlier, later in pairwise(sizes)):
            raise InvalidTrainingMixtureError(
                f"the vocabularies of a mixture do not shrink along it, got {sizes}"
            )

    @property
    def name(self) -> str:
        """What the mixture is called: the corpora's names, joined in order."""
        return "+".join(corpus.name for corpus in self.corpora)

    @property
    def vocabulary_size(self) -> int:
        """Channels the mixture's vocabulary holds: the last corpus's, which continues the rest."""
        return self.corpora[-1].vocabulary_size
