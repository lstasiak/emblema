import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.identifiers import UnitKey


@dataclass(frozen=True)
class UnitSplit:
    """Which units of a corpus train a scheme and which only validate it.

    The split is made on units, before any window is cut, so no two windows of one unit fall on
    different sides of it; splitting windows would put neighbours that overlap on both sides and
    leak. Statistics of a tokenisation scheme are fitted on the training side only.

    Invariants: both sides hold at least one unit and share none.

    Attributes:
        training: Units whose data fits the scheme and trains the model.
        validation: Units held out from fitting.
    """

    training: frozenset[UnitKey]
    validation: frozenset[UnitKey]

    def __post_init__(self) -> None:
        if not self.training or not self.validation:
            raise InvalidUnitSplitError("both sides of a split must hold at least one unit")
        shared = sorted(str(key) for key in self.training & self.validation)
        if shared:
            raise InvalidUnitSplitError(f"units on both sides of the split: {shared}")

    @classmethod
    def by_seed(cls, keys: Iterable[UnitKey], validation_fraction: float, seed: int) -> Self:
        """A reproducible split: the same units, fraction and seed always give the same sides.

        Units are sorted by key before a seeded shuffle, so the result does not depend on the
        order the reader delivered them in. The validation side takes the whole units that fit
        the fraction, rounded down.

        Raises:
            InvalidUnitSplitError: If a key repeats, the fraction is not strictly between 0 and 1,
                or too few units are given for the fraction to leave a unit on each side.
        """
        ordered = sorted(keys, key=str)
        if len(set(ordered)) != len(ordered):
            raise InvalidUnitSplitError("unit keys must be unique")
        if not 0.0 < validation_fraction < 1.0:
            raise InvalidUnitSplitError(
                f"validation fraction must lie strictly between 0 and 1, got {validation_fraction}"
            )
        held_out = int(len(ordered) * validation_fraction)
        if held_out == 0:
            raise InvalidUnitSplitError(
                f"{len(ordered)} units at fraction {validation_fraction} leave no unit for "
                f"validation"
            )
        random.Random(seed).shuffle(ordered)
        return cls(training=frozenset(ordered[held_out:]), validation=frozenset(ordered[:held_out]))
