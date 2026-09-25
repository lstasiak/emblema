from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from emblema.evaluation.domain.exceptions import InvalidFamilyCorrectionError


@dataclass(frozen=True, kw_only=True)
class BenjaminiHochbergCorrection:
    """Which of a family of comparisons is rejected so that the false discoveries stay a share.

    The step-up procedure of Benjamini and Hochberg (1995). What it controls is the expected
    share of false discoveries among the rejections, not the chance of any false discovery, so
    it rejects more than Holm on the same family and promises less about each rejection; the
    share holds under positive dependence, which cells of one validation side have.

    Invariants: the level lies strictly between zero and one.

    Attributes:
        alpha: The share of false discoveries the family is held to.
    """

    alpha: float = 0.05

    NAME: ClassVar[str] = "benjamini_hochberg"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise InvalidFamilyCorrectionError(f"alpha must lie in (0, 1), got {self.alpha}")

    def rejected(
        self, p_values: Sequence[float], *, family_size: int | None = None
    ) -> tuple[bool, ...]:
        """Whether each comparison is rejected, in the order the p-values were given.

        A ``family_size`` larger than the p-values given names a family whose other members
        have not been measured: each enters with a p-value of one, which ranks last and is
        never rejected, while the measured ones are held to their rank over the whole family,
        so an incomplete family is read more strictly than the complete one and never less.

        Raises:
            InvalidFamilyCorrectionError: If the family is empty or smaller than the p-values
                given, or a p-value lies outside ``[0, 1]``.
        """
        if not p_values:
            raise InvalidFamilyCorrectionError("a family needs at least one comparison")
        for p_value in p_values:
            if not 0.0 <= p_value <= 1.0:
                raise InvalidFamilyCorrectionError(f"a p-value must lie in [0, 1], got {p_value}")
        family = len(p_values) if family_size is None else family_size
        if family < len(p_values):
            raise InvalidFamilyCorrectionError(
                f"a family of {family} cannot hold {len(p_values)} comparisons"
            )
        ranked = sorted(range(len(p_values)), key=lambda index: p_values[index])
        largest_holding = -1
        for rank, index in enumerate(ranked):
            if p_values[index] <= self.alpha * (rank + 1) / family:
                largest_holding = rank
        verdicts = [False] * len(p_values)
        for rank in range(largest_holding + 1):
            verdicts[ranked[rank]] = True
        return tuple(verdicts)
