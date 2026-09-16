from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidTaskSplitError
from emblema.evaluation.domain.identifiers import UnitKey


@dataclass(frozen=True)
class FrozenTestSplit:
    """The units a task is finally scored on, named when the task is created and untouched until.

    The units may not be in any published corpus yet, and for a task whose source publishes an
    official test set they usually are not: pulling them into the corpus a backbone pretrains on
    would let the model read them without a label, which is the leak this side exists to prevent.
    So the side carries where the units come from as well as what they are called, and whoever
    materialises them later knows what to publish.

    The keys are a field because the split checks them against its other sides and a repository
    stores them. The rule that decides who may act on them is not here but on the task, which
    knows what a run is for; a caller that reaches around it is not prevented by the type, it is
    contradicted by the operation it skipped.

    Invariants: at least one unit; the source is non-blank without surrounding whitespace.

    Attributes:
        units: Units held out for the final run.
        source: Where they come from, as the set is known outside this system.
    """

    units: frozenset[UnitKey]
    source: str

    def __post_init__(self) -> None:
        if not self.units:
            raise InvalidTaskSplitError("a frozen test side must hold at least one unit")
        if not self.source or self.source != self.source.strip():
            raise InvalidTaskSplitError(
                "the source of a frozen test side must be non-blank without surrounding whitespace"
            )
