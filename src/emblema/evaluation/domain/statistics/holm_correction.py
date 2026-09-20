from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidHolmCorrectionError


@dataclass(frozen=True, kw_only=True)
class HolmCorrection:
    """Which of a family of comparisons is rejected at a level shared by the whole family.

    Step-down: the p-values are ranked, the smallest is held to the level over the family's
    size, the next to the level over one fewer, and so on; the first that fails stops the
    procedure and everything ranked after it stands. The family-wise error stays at the level
    whatever the comparisons' dependence, which is what a family of cells on one validation side
    needs.

    Invariants: the level lies strictly between zero and one.

    Attributes:
        alpha: The level the family is tested at.
    """

    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise InvalidHolmCorrectionError(f"alpha must lie in (0, 1), got {self.alpha}")

    def rejected(
        self, p_values: Sequence[float], *, family_size: int | None = None
    ) -> tuple[bool, ...]:
        """Whether each comparison is rejected, in the order the p-values were given.

        A ``family_size`` larger than the p-values given names a family whose other members
        have not been measured: each enters with a p-value of one, which is never rejected and
        holds the measured ones to the levels the whole family sets, so an incomplete family is
        read more strictly than the complete one and never less.

        Raises:
            InvalidHolmCorrectionError: If the family is empty or smaller than the p-values
                given, or a p-value lies outside ``[0, 1]``.
        """
        if not p_values:
            raise InvalidHolmCorrectionError("a family needs at least one comparison")
        for p_value in p_values:
            if not 0.0 <= p_value <= 1.0:
                raise InvalidHolmCorrectionError(f"a p-value must lie in [0, 1], got {p_value}")
        family = len(p_values) if family_size is None else family_size
        if family < len(p_values):
            raise InvalidHolmCorrectionError(
                f"a family of {family} cannot hold {len(p_values)} comparisons"
            )
        ranked = sorted(range(len(p_values)), key=lambda index: p_values[index])
        verdicts = [False] * len(p_values)
        for rank, index in enumerate(ranked):
            if p_values[index] > self.alpha / (family - rank):
                break
            verdicts[index] = True
        return tuple(verdicts)
