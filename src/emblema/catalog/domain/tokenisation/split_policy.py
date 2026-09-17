"""How a publication chooses the units it holds out: by drawing them, or by naming them.

Two classes and the name of their union, because they are the two halves of one decision and
neither is a special case of the other. A policy is what a publication asked for; ``UnitSplit``
is what it got.

A seeded fraction is right where the units differ only in the draw: every unit is one realisation
of the same process, so which of them is held out says nothing. It is wrong where they differ in
kind — the months in which an instrument was still settling are not the months that follow, and a
draw that puts them all on one side leaves a held-out side no model trained on the other reaches.
Such a corpus names its held-out units, and their names travel with the publication in place of a
seed, which nobody could invert into them.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit


@dataclass(frozen=True)
class SeededSplit:
    """Hold out the share of the units a seed draws.

    Attributes:
        validation_fraction: Share of units drawn onto the held-out side.
        seed: Seed the draw is made with, and what the publication records of it.
    """

    validation_fraction: float
    seed: int

    def applied_to(self, keys: Sequence[UnitKey]) -> UnitSplit:
        """The split this draw makes of those units.

        Raises:
            InvalidUnitSplitError: If a key repeats, the fraction is not strictly between 0 and
                1, or too few units are given for it to hold one out.
        """
        return UnitSplit.by_seed(keys, self.validation_fraction, self.seed)

    @property
    def recorded_seed(self) -> int | None:
        """The seed a manifest records, which a drawn split has and a named one has not."""
        return self.seed


@dataclass(frozen=True)
class NamedSplit:
    """Hold out exactly the units named, whatever a draw would have done.

    Attributes:
        held_out: The units to hold out; recorded by the publication in place of a seed.
    """

    held_out: tuple[UnitKey, ...]

    @classmethod
    def of(cls, held_out: Iterable[UnitKey]) -> "NamedSplit":
        return cls(tuple(held_out))

    def applied_to(self, keys: Sequence[UnitKey]) -> UnitSplit:
        """The split naming these units makes of those units.

        Raises:
            InvalidUnitSplitError: If a key repeats, a named unit is not among them, or a side
                would be left empty.
        """
        return UnitSplit.of_held_out(keys, self.held_out)

    @property
    def recorded_seed(self) -> int | None:
        """Nothing: a named split is recorded by its names, and no seed produced it."""
        return None


SplitPolicy = SeededSplit | NamedSplit
