from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidHeadPoolingError, UnknownKnobError
from emblema.evaluation.domain.tuning.knob import turned


class PoolingScheme(StrEnum):
    """How the states of a window's tokens become the one state a head reads.

    Every scheme here is free of the channel layout: it weighs tokens by where they sit in the
    window or by what they hold, never by which channel they came from, so a head under any of
    them reads a corpus of eight channels as it reads one of forty-four. A pooling laid out per
    channel would be as wide as the corpus and is left to the classical baselines.

    Attributes:
        MEAN: The mean over every observed token of the window.
        TAIL: The mean over the observed tokens in the last share of the window, the window's
            static features included, since a static feature is current at every instant.
        ATTENTION: A weighted mean whose weights a learnt query reads off the states themselves,
            so the head can learn which part of the window to read from.
    """

    MEAN = "mean"
    TAIL = "tail"
    ATTENTION = "attention"

    @property
    def learns_weights(self) -> bool:
        """Whether the pooling has weights of its own that a run trains."""
        return self is PoolingScheme.ATTENTION


class StaticsPlacement(StrEnum):
    """Where a window's static features stand in the state a head reads.

    A static feature (age, sex, the ward) is one token against the hundreds of readings a stay
    holds, so in the pooled state it weighs as one reading among them. Set apart, the readings
    are pooled by the scheme and the static features by their own mean, and the head reads the
    two side by side, as networks published on these stays read their demographics. Neither
    placement reads the channel layout: a window without static features gives zeros on their
    side.

    Attributes:
        AMONG: Pooled with the readings, as every network was before this was a knob.
        APART: Pooled on their own and set beside the readings, doubling the state's width.
    """

    AMONG = "among"
    APART = "apart"


@dataclass(frozen=True, kw_only=True)
class HeadPooling:
    """The pooling of a network's head: which scheme, what a tail keeps, where statics stand.

    Stated once for every network a campaign compares, so a set encoder and a patch model pool
    by one vocabulary and a variant name turns the same knobs on both. A share under a scheme
    that reads none would be a knob recorded as turned and changing nothing, so it is refused
    rather than normalised. A tail over the whole window computes the mean and is allowed: a
    variant turns one knob at a time, and the tail has to exist before its share is set. Set
    apart, the static features leave whichever scheme pools the readings, a tail included.

    Invariants: the share lies in ``(0, 1]``, and is one unless the scheme is a tail.

    Attributes:
        pooling: Which scheme.
        tail_share: Share of the window's length a tail keeps, from its end; one otherwise.
        statics: Where the static features stand: among the readings unless a variant turns it.
    """

    pooling: PoolingScheme
    tail_share: float = 1.0
    statics: StaticsPlacement = StaticsPlacement.AMONG

    KNOBS: ClassVar[tuple[str, ...]] = ("pooling", "tail_share", "statics")

    def __post_init__(self) -> None:
        if not isfinite(self.tail_share) or not 0.0 < self.tail_share <= 1.0:
            raise InvalidHeadPoolingError(f"tail_share must lie in (0, 1], got {self.tail_share}")
        if self.tail_share < 1.0 and self.pooling is not PoolingScheme.TAIL:
            raise InvalidHeadPoolingError(
                f"{self.pooling} reads the whole window, so a share of it is no knob"
            )

    @classmethod
    def mean(cls) -> Self:
        """The pooling every network started with: the mean over the window."""
        return cls(pooling=PoolingScheme.MEAN)

    def tuned(self, knob: str, value: str) -> Self:
        """This pooling with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the pooling has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"a pooling has no knob {knob!r}; it turns {self.KNOBS}")
        return turned(self, knob, value)

    @property
    def width_factor(self) -> int:
        """How many states of the encoder's width the pooled state is made of."""
        return 2 if self.statics is StaticsPlacement.APART else 1

    def parameters(self) -> dict[str, str | float]:
        """The scheme and its share flattened to scalars, in a fixed order.

        The placement of the static features is not among them: a candidate's description is
        checked whole against the one its campaign stored, and those stored before the placement
        was a knob name the scheme and the share alone (``turned_away``).
        """
        return {"pooling": str(self.pooling), "tail_share": float(self.tail_share)}

    def turned_away(self) -> dict[str, str]:
        """The placement of the static features, named only where a variant turned it."""
        if self.statics is StaticsPlacement.AMONG:
            return {}
        return {"statics": str(self.statics)}
