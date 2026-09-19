"""How a publication chooses the units it holds out: by drawing them, naming them, or by part.

Three classes and the name of their union, because they are the halves of one decision and none
is a special case of another. A policy is what a publication asked for; ``UnitSplit`` is what it
got.

A seeded fraction is right where the units differ only in the draw: every unit is one realisation
of the same process, so which of them is held out says nothing. It is wrong where they differ in
kind — the months in which an instrument was still settling are not the months that follow, and a
draw that puts them all on one side leaves a held-out side no model trained on the other reaches.
Such a corpus names its held-out units, and their names travel with the publication in place of a
seed, which nobody could invert into them.

Where the publisher of a corpus already drew that line, the units are named a part at a time: a
challenge that hands out one set to train on and another to validate against has said which units
differ in kind, and repeating its division is what makes a number comparable with the numbers
published under it.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidUnitSplitError
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


@dataclass(frozen=True)
class SubsetSplit:
    """Hold out every unit of one part of the corpus, whatever a draw would have done.

    What naming the units says, for a corpus whose parts already say it and whose units are too
    many to name: a challenge's validation set is thousands of keys, and a publication that
    repeated them on its command line would state the same division less legibly.

    Attributes:
        subset: Part of the corpus whose units are held out; recorded by the publication in place
            of a seed.
    """

    subset: str

    def __post_init__(self) -> None:
        if not self.subset or self.subset != self.subset.strip():
            raise InvalidUnitSplitError(
                "the part held out must be non-blank without surrounding whitespace"
            )

    def applied_to(self, keys: Sequence[UnitKey]) -> UnitSplit:
        """The split that part makes of those units.

        Raises:
            InvalidUnitSplitError: If a key repeats, no unit was read from that part, or a side
                would be left empty.
        """
        held_out = [key for key in keys if key.belongs_to(self.subset)]
        if not held_out:
            raise InvalidUnitSplitError(f"no unit of the corpus was read from {self.subset}")
        return UnitSplit.of_held_out(keys, held_out)

    @property
    def recorded_seed(self) -> int | None:
        """Nothing: the part is what was asked for, and no seed produced this split."""
        return None


SplitPolicy = SeededSplit | NamedSplit | SubsetSplit
