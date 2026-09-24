from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidInnerHoldoutError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.shared.kernel.ordering import seeded_rank


@dataclass(frozen=True, kw_only=True)
class InnerHoldout:
    """How the tuning side is divided when a run chooses among variants rather than compares.

    Choosing a variant on the side a comparison is later read from would leave the winner's
    number there optimistic by exactly the luck that made it win. So a selection run learns
    from part of the tuning units and is scored on the rest, and the validation side is never
    read. Each repeat divides the units afresh under its own seed — repeated random holdout,
    whose overlapping divisions the selection rule corrects its spread for.

    The units are ranked under the seed and one in every ``one_in`` of them, rounded up, is held
    out from the top of that ranking, so the same units always divide the same way whatever
    order they are named in.

    Invariants: one in at least two units is held out, so some are left to learn from.

    Attributes:
        one_in: One tuning unit in this many is held out to score on.
    """

    one_in: int

    def __post_init__(self) -> None:
        if self.one_in < 2:
            raise InvalidInnerHoldoutError(f"one_in must be at least two, got {self.one_in}")

    @property
    def test_to_train(self) -> float:
        """How many units are scored for every one learnt from, for the selection rule."""
        return 1.0 / (self.one_in - 1)

    def divided(
        self, units: frozenset[UnitKey], seed: int
    ) -> tuple[frozenset[UnitKey], frozenset[UnitKey]]:
        """The units to learn from and the units to score on, under ``seed``.

        Raises:
            InvalidInnerHoldoutError: If there are too few units to leave one on either side.
        """
        ranked = sorted(units, key=lambda unit: seeded_rank(seed, "inner-holdout", str(unit)))
        held = -(-len(ranked) // self.one_in)
        if held >= len(ranked):
            raise InvalidInnerHoldoutError(
                f"{len(ranked)} tuning units leave none to learn from when one in "
                f"{self.one_in} is held out"
            )
        return frozenset(ranked[held:]), frozenset(ranked[:held])
