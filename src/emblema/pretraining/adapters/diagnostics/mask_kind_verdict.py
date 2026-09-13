from dataclasses import dataclass

from emblema.pretraining.domain.mask_kind import MaskKind


@dataclass(frozen=True)
class MaskKindVerdict:
    """The model against the trivial baseline of one kind of mask, over the tokens it hid.

    The claim a kind of mask teaches something is the claim the model beats the answer anyone
    could have written down; ``excess`` is by how much, in the units of the loss, and ``trivial``
    is the verdict. Channels reported apart — constant ones, timeless ones — have their own rows,
    because a value that never changes is recovered by every method and would flatter the model
    and the baseline alike.

    Attributes:
        kind: The kind of mask the tokens were hidden by.
        apart: Whether this row is the channels reported apart from the verdict.
        tokens: How many hidden tokens the errors are averaged over.
        model_error: Mean squared error of the model on them.
        baseline_error: Mean squared error of the baseline matched to the kind.
    """

    kind: MaskKind
    apart: bool
    tokens: int
    model_error: float
    baseline_error: float

    @property
    def excess(self) -> float:
        """How much lower the model's error is than the baseline's; negative when it is higher."""
        return self.baseline_error - self.model_error

    @property
    def trivial(self) -> bool:
        """Whether the baseline does as well as the model, so the kind taught nothing it shows."""
        return self.tokens > 0 and self.excess <= 0.0
