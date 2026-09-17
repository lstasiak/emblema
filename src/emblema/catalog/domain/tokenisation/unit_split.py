from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.ordering import seeded_rank


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

        Each unit is placed by ``seeded_rank`` of its own key, and the validation side takes the
        whole units that fit the fraction, rounded down. Comparing two versions of a corpus is the
        point of the project, so a split that reshuffled when a unit was added would quietly take
        that away; ranking each unit on its own is what stops it, and the key is broken by the
        unit's own name so that two units ranking equal still land in a fixed order.

        Raises:
            InvalidUnitSplitError: If a key repeats, the fraction is not strictly between 0 and 1,
                or too few units are given for the fraction to hold one out for validation.
        """
        units = list(keys)
        if len(set(units)) != len(units):
            raise InvalidUnitSplitError("unit keys must be unique")
        if not 0.0 < validation_fraction < 1.0:
            raise InvalidUnitSplitError(
                f"validation fraction must lie strictly between 0 and 1, got {validation_fraction}"
            )
        held_out = int(len(units) * validation_fraction)
        if held_out == 0:
            raise InvalidUnitSplitError(
                f"{len(units)} units at fraction {validation_fraction} leave no unit for validation"
            )
        ordered = sorted(units, key=lambda key: (seeded_rank(seed, key), str(key)))
        return cls(training=frozenset(ordered[held_out:]), validation=frozenset(ordered[:held_out]))

    @classmethod
    def of_held_out(cls, keys: Iterable[UnitKey], held_out: Iterable[UnitKey]) -> Self:
        """The split that holds out exactly the units named, and keeps the rest.

        What a seeded draw cannot do: put particular units on the held-out side. A corpus whose
        units differ in kind rather than only in draw — months of a mission in which the
        instrument was still settling, say — is split by naming them, and the names travel with
        the publication instead of a seed nobody could invert.

        Raises:
            InvalidUnitSplitError: If a key repeats on either side, a unit named is not a unit of
                the corpus, or a side is left empty.
        """
        units = list(keys)
        if len(set(units)) != len(units):
            raise InvalidUnitSplitError("unit keys must be unique")
        wanted = list(held_out)
        if len(set(wanted)) != len(wanted):
            raise InvalidUnitSplitError("held-out keys must be unique")
        unknown = sorted(str(key) for key in set(wanted) - set(units))
        if unknown:
            raise InvalidUnitSplitError(f"units held out that the corpus does not hold: {unknown}")
        return cls(training=frozenset(set(units) - set(wanted)), validation=frozenset(wanted))
