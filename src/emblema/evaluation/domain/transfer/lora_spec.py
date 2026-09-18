from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidLoraSpecError


@dataclass(frozen=True, kw_only=True)
class LoraSpec:
    """The low-rank updates a transfer adds beside the frozen layers of the backbone.

    Each named linear layer keeps its weights and gains a product of two thin matrices whose
    rank bounds what the update can express; the scale sets how much of it reaches the output.
    Which layers are named is a parameter of the method like the rank: updating the attention
    alone and updating the feed-forward network too are two different caps on the degrees of
    freedom, and a reader of the curve has to know which one was measured.

    Invariants: the rank is positive; the scale is positive and finite; the dropout lies in
    ``[0, 1)``; at least one layer is named, each name non-blank without surrounding whitespace,
    none twice.

    Attributes:
        rank: Inner size of the update.
        alpha: Scale of the update; the product is multiplied by ``alpha / rank``.
        dropout: Dropout applied to the update's input while training.
        targets: Which linear layers the update is added beside, each as a dotted run of
            consecutive segments of the layer's path in the backbone.
    """

    rank: int
    alpha: float
    dropout: float
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise InvalidLoraSpecError(f"rank must be positive, got {self.rank}")
        if not isfinite(self.alpha) or self.alpha <= 0.0:
            raise InvalidLoraSpecError(f"alpha must be positive and finite, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise InvalidLoraSpecError(f"dropout must lie in [0, 1), got {self.dropout}")
        if not self.targets:
            raise InvalidLoraSpecError("a low-rank update must name at least one layer")
        for target in self.targets:
            if not target or target != target.strip():
                raise InvalidLoraSpecError(
                    "a target must be non-blank without surrounding whitespace"
                )
        if len(set(self.targets)) != len(self.targets):
            raise InvalidLoraSpecError(f"a target is named twice: {list(self.targets)}")

    @property
    def scaling(self) -> float:
        """What the product of the two matrices is multiplied by before it is added."""
        return self.alpha / self.rank
