import math
from dataclasses import dataclass
from itertools import pairwise

from emblema.catalog.domain.exceptions import InvalidValueBoundsError


@dataclass(frozen=True, kw_only=True)
class ValueBounds:
    """What a channel's value may be: removed outside an outer pair of bounds, clipped inside it.

    A value outside the outer pair is one no unit can have — a pH of 735, a temperature of
    minus 18 °C — and is removed, so it becomes a value never measured. A value between an outer and
    an inner bound is possible but extreme and is clipped to the inner one. Bounds are in the
    unit the corpus records the channel in and are fixed before any data is seen.

    Invariants: the channel is named without surrounding whitespace; every bound is finite;
    removed below ≤ clipped below ≤ clipped above ≤ removed above.

    Attributes:
        channel: Name of the channel within its corpus.
        removed_below: A value below it is removed.
        clipped_below: A value below it, and not removed, is raised to it.
        clipped_above: A value above it, and not removed, is lowered to it.
        removed_above: A value above it is removed.
    """

    channel: str
    removed_below: float
    clipped_below: float
    clipped_above: float
    removed_above: float

    def __post_init__(self) -> None:
        if not self.channel or self.channel != self.channel.strip():
            raise InvalidValueBoundsError(
                "channel name must be non-blank without surrounding whitespace"
            )
        bounds = (self.removed_below, self.clipped_below, self.clipped_above, self.removed_above)
        if not all(math.isfinite(bound) for bound in bounds):
            raise InvalidValueBoundsError(f"bounds of {self.channel!r} must be finite: {bounds}")
        if any(later < earlier for earlier, later in pairwise(bounds)):
            raise InvalidValueBoundsError(
                f"bounds of {self.channel!r} must run removed, clipped, clipped, removed from "
                f"low to high: {bounds}"
            )

    def cleaned(self, value: float) -> float | None:
        """The value as the bounds allow it; ``None`` where it is removed."""
        if value < self.removed_below or value > self.removed_above:
            return None
        return min(max(value, self.clipped_below), self.clipped_above)
