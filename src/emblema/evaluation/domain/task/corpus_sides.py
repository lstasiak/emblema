from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.identifiers import UnitKey


@dataclass(frozen=True, kw_only=True)
class CorpusSides:
    """How a published corpus already divides its units, and the name it goes by.

    A task inherits this division rather than drawing its own. The corpus fitted the statistics
    that normalise every token on its training side, and a backbone pretrained on that side has
    read those units without their labels; a task that validated on them would be measuring a
    model on data it has already seen. The units held out by the corpus are the ones no part of
    the pipeline has touched, so they are the ones worth validating on.

    Invariants: both sides hold at least one unit and share none; the name is non-blank without
    surrounding whitespace.

    Attributes:
        corpus: Name the corpus was published under.
        training: Units whose data fitted the statistics and fed the pretraining.
        validation: Units the corpus held out.
    """

    corpus: str
    training: frozenset[UnitKey]
    validation: frozenset[UnitKey]

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidTaskSplitError(
                "corpus name must be non-blank without surrounding whitespace"
            )
        if not self.training or not self.validation:
            raise InvalidTaskSplitError("both sides of a published corpus must hold a unit")
        shared = sorted(str(unit) for unit in self.training & self.validation)
        if shared:
            raise InvalidTaskSplitError(f"units on both sides of the published corpus: {shared}")
