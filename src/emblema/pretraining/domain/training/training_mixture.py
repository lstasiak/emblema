from dataclasses import dataclass
from itertools import pairwise
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidTrainingMixtureError
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture_shape import TrainingMixtureShape


@dataclass(frozen=True, kw_only=True)
class TrainingMixture:
    """The corpora one run learns from together, under one vocabulary chained through them.

    A run over several corpora reads each from its own publication, and what makes them one
    corpus to the model is the vocabulary: each corpus after the first was published continuing
    the vocabulary of the one before, so its channels take identifiers no earlier corpus took.
    The mixture holds the corpora in that order and checks the chain by the channel names each
    publication carries, because two corpora published apart would both carry identifiers from
    one, and a run over them would learn a collision without an error. A run over one corpus is
    a mixture of one.

    Invariants: at least one corpus; no corpus twice; each corpus's channel names are a prefix
    of the next's. A corpus left out of the middle of a chain keeps the chain a chain — the names
    before the gap are a prefix of the names after it — so a mixture that leaves a corpus out
    reads the same publications with one fewer.

    Attributes:
        corpora: The corpora, in the order their vocabulary was chained.
    """

    corpora: tuple[TrainingCorpus, ...]

    def __post_init__(self) -> None:
        # Building the shape runs its invariants: a mixture whose shape is not one is none.
        _ = self.shape
        for earlier, later in pairwise(self.corpora):
            if not later.continues(earlier):
                raise InvalidTrainingMixtureError(
                    f"the vocabulary of {later.name!r} does not continue the vocabulary of "
                    f"{earlier.name!r}: a mixture reads corpora published one after another"
                )

    @classmethod
    def of(cls, *corpora: TrainingCorpus) -> Self:
        """The mixture of these corpora, in this order."""
        return cls(corpora=corpora)

    @property
    def name(self) -> str:
        """What the mixture is called: the corpora's names, joined in order."""
        return self.shape.name

    @property
    def channels(self) -> tuple[str, ...]:
        """The vocabulary the run trains under: the last corpus's, which continues the rest."""
        return self.corpora[-1].channels

    @property
    def vocabulary_size(self) -> int:
        return len(self.channels)

    @property
    def shape(self) -> TrainingMixtureShape:
        """The mixture without its windows: what a run's signature is computed from.

        Raises:
            InvalidTrainingMixtureError: If no corpus is given, one is given twice or the
                vocabularies shrink along the order.
        """
        return TrainingMixtureShape(corpora=tuple(corpus.shape for corpus in self.corpora))
