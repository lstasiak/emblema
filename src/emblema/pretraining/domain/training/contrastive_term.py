from dataclasses import dataclass
from math import isfinite, log
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidContrastiveTermError


@dataclass(frozen=True, kw_only=True)
class ContrastiveTerm:
    """A second term of the objective: the two halves of a masked window told apart from others.

    Beside the loss at the hidden tokens, a run may ask the encoder to recognise a window: the
    tokens it saw and the tokens hidden from it, each summarised to one point, are a pair, and
    every other window of the batch is a distractor. The term is a parameter of the method like
    the loss and the target: a run that trains it is a different run over the same data, so it
    is stated by the experiment, reported with the run and digested into its signature. A run
    that states no term renders nothing, as every run before the term existed.

    The arithmetic over tensors belongs to the objective's adapter; what is stated here is what
    it is held to: the term's weight against one hidden token's loss, the temperature the
    similarities are divided by, and what the term costs a model that tells nothing apart.

    Invariants: the weight is finite and not negative, and a stated term has a positive, finite
    temperature; a term of no weight states no temperature, since a knob nobody reads would tell
    two identical runs apart in the signature.

    Attributes:
        weight: What one anchor's contrastive loss weighs against one hidden token's loss; zero
            is no term.
        temperature: What the cosine similarities are divided by before the softmax; zero where
            there is no term.
    """

    weight: float = 0.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.weight) or self.weight < 0.0:
            raise InvalidContrastiveTermError(
                f"weight must be finite and not negative, got {self.weight}"
            )
        if self.weight == 0.0:
            if self.temperature != 0.0:
                raise InvalidContrastiveTermError(
                    f"a term of no weight reads no temperature, got {self.temperature}"
                )
            return
        if not isfinite(self.temperature) or self.temperature <= 0.0:
            raise InvalidContrastiveTermError(
                f"temperature must be positive and finite, got {self.temperature}"
            )

    @classmethod
    def none(cls) -> Self:
        """No contrast: the objective every run stated before the term existed."""
        return cls()

    @classmethod
    def of(cls, *, weight: float, temperature: float) -> Self:
        """A term of this weight at this temperature."""
        return cls(weight=weight, temperature=temperature)

    @property
    def is_stated(self) -> bool:
        return self.weight > 0.0

    @staticmethod
    def chance_of(windows: int) -> float:
        """What one anchor costs a model whose similarities are all equal, over ``windows`` pairs.

        The definition the objective's adapter is held to. Each of the two halves of each window
        is an anchor; it has one positive, the other half of its window, among the halves of
        every window but itself, so the softmax over equal similarities gives its positive one
        part in ``2·windows - 1``. One window alone has no distractor and costs nothing.

        Raises:
            InvalidContrastiveTermError: If the count is negative.
        """
        if windows < 0:
            raise InvalidContrastiveTermError(f"no batch holds {windows} windows")
        if windows < 2:
            return 0.0
        return log(2 * windows - 1)
