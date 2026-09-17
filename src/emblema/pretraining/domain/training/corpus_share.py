from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite

from emblema.pretraining.domain.exceptions import InvalidCorpusShareError
from emblema.shared.kernel.ordering import seeded_rank

# Frames the ranking apart from the one the Catalog splits units with, so a run whose seed equals
# the split's does not read the units that happen to rank just past the validation cut.
_RANKING = "training-share"


@dataclass(frozen=True, kw_only=True)
class CorpusShare:
    """How much of a corpus's training side a run reads, and which units that is.

    A run over part of a corpus takes whole units, never a share of each unit's windows: windows
    of one unit overlap, and a cut through them would put neighbours on both sides of it, as a
    split through windows would. The units are ranked by the run's seed, each on its own, and the
    share takes the lowest ranks that fit the fraction, rounded up so that a share of one unit is
    one unit. Two shares of one seed are therefore nested — a tenth lies inside a quarter, a
    quarter inside a half — which is what a curve over fractions wants: each point adds data to the
    last rather than drawing afresh. The validation side is never shared: a run over a tenth is
    scored on all of it, or its loss would not be comparable with a run over the whole.

    Invariants: the fraction lies in ``(0, 1]``.

    Attributes:
        fraction: Share of the training units read.
        seed: Seed the units are ranked by; the run's own.
    """

    fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise InvalidCorpusShareError(f"fraction must lie in (0, 1], got {self.fraction}")

    @property
    def is_whole(self) -> bool:
        return self.fraction == 1.0

    def select(self, units: Sequence[str]) -> tuple[str, ...]:
        """The units this share reads, in the order they were given.

        Raises:
            InvalidCorpusShareError: If a unit repeats.
        """
        if len(set(units)) != len(units):
            raise InvalidCorpusShareError("unit keys must be unique")
        if self.is_whole:
            return tuple(units)
        ranked = sorted(units, key=lambda unit: (seeded_rank(self.seed, _RANKING, unit), unit))
        # Rounded before the ceiling: three tenths of ten units is three, not the four a binary
        # 0.3 × 10 would round up to.
        chosen = set(ranked[: ceil(round(self.fraction * len(units), 9))])
        return tuple(unit for unit in units if unit in chosen)
