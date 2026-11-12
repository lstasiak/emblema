import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist
from typing import Final

from emblema.catalog.domain.channels.normalisation import Normalisation
from emblema.catalog.domain.exceptions import InvalidChannelStatisticsError

# The median absolute deviation times this is the standard deviation of a normal distribution,
# so a robust deviation and a standard one measure the bulk of a normal channel alike.
MAD_TO_DEVIATION: Final = 1.4826
# How far from the median a robust score may lie unless the publication names another bound:
# past it a value is an extreme of the channel, and how far past says more about the channel's
# tail than about the unit.
ROBUST_BOUND: Final = 10.0
# Knots of the quantile score: the training values' quantiles at every thousandth of rank.
QUANTILE_KNOTS: Final = 1001
# The rank a value beyond every training value takes, so its normal score stays finite (about
# ±5.2) rather than the infinity the rank of nought or one has.
RANK_FLOOR: Final = 1e-7
_NORMAL: Final = NormalDist()


@dataclass(frozen=True)
class ChannelStatistics:
    """What the training data established about one channel, and the normalisation it implies.

    The statistics are facts about the data a scheme was fitted on, kept as such rather than
    folded into a transform, because they are also the provenance of the model and the only
    description of a channel the data itself offers. Count, mean and standard deviation are
    always kept; the median and its absolute deviation, or the quantiles, only where the
    normalisation reads them, so the facts a manifest states are the ones its values were made
    from. A channel whose deviation vanished in training measures departures in raw units
    instead, so a value that does depart later stays finite and honest about how far it strayed.

    Invariants: the count is positive; every figure is finite; the deviations are not negative;
    a robust normalisation has a median, its deviation and a positive bound and no quantiles, a
    quantile one has ``QUANTILE_KNOTS`` quantiles in order and none of those, a standard one has
    neither.

    Attributes:
        count: Values the statistics were computed over.
        mean: Arithmetic mean.
        std: Population standard deviation.
        normalisation: How a value of the channel is normalised.
        median: Median, where the normalisation is robust.
        mad: Median absolute deviation from the median, where the normalisation is robust.
        quantiles: The values at every ``1 / (QUANTILE_KNOTS - 1)`` of rank, lowest first, where
            the normalisation is by quantile.
        bound: How far from the median a robust score may lie, in robust deviations, where the
            normalisation is robust; a publication chooses it for the tails its channels have.
    """

    count: int
    mean: float
    std: float
    normalisation: Normalisation = Normalisation.STANDARD
    median: float | None = None
    mad: float | None = None
    quantiles: tuple[float, ...] = ()
    bound: float | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise InvalidChannelStatisticsError(f"count must be positive, got {self.count}")
        if not math.isfinite(self.mean):
            raise InvalidChannelStatisticsError(f"mean must be finite, got {self.mean}")
        if not (math.isfinite(self.std) and self.std >= 0):
            raise InvalidChannelStatisticsError(
                f"std must be finite and non-negative, got {self.std}"
            )
        robust = self.normalisation is Normalisation.ROBUST
        if (self.median is not None or self.mad is not None) != robust:
            raise InvalidChannelStatisticsError(
                f"a median and its deviation belong to a robust normalisation, not a "
                f"{self.normalisation} one"
            )
        if (self.bound is not None) != robust:
            raise InvalidChannelStatisticsError(
                f"a bound belongs to a robust normalisation, not a {self.normalisation} one"
            )
        if self.bound is not None and not (math.isfinite(self.bound) and self.bound > 0):
            raise InvalidChannelStatisticsError(
                f"bound must be finite and positive, got {self.bound}"
            )
        if self.median is not None and not math.isfinite(self.median):
            raise InvalidChannelStatisticsError(f"median must be finite, got {self.median}")
        if self.mad is not None and not (math.isfinite(self.mad) and self.mad >= 0):
            raise InvalidChannelStatisticsError(
                f"mad must be finite and non-negative, got {self.mad}"
            )
        by_quantile = self.normalisation is Normalisation.QUANTILE
        if bool(self.quantiles) != by_quantile:
            raise InvalidChannelStatisticsError(
                f"quantiles belong to a quantile normalisation, not a {self.normalisation} one"
            )
        if by_quantile:
            self._check_quantiles()

    def _check_quantiles(self) -> None:
        if len(self.quantiles) != QUANTILE_KNOTS:
            raise InvalidChannelStatisticsError(
                f"a quantile normalisation keeps {QUANTILE_KNOTS} quantiles, got "
                f"{len(self.quantiles)}"
            )
        if not all(math.isfinite(value) for value in self.quantiles):
            raise InvalidChannelStatisticsError("quantiles must be finite")
        if any(later < earlier for earlier, later in pairwise(self.quantiles)):
            raise InvalidChannelStatisticsError("quantiles must not decrease")

    def normalise(self, value: float) -> float:
        match self.normalisation:
            case Normalisation.STANDARD:
                return (value - self.mean) / self._standard_scale
            case Normalisation.ROBUST:
                centre, scale, bound = self._robust_frame
                return min(max((value - centre) / scale, -bound), bound)
            case Normalisation.QUANTILE:
                rank = min(max(self._rank(value), RANK_FLOOR), 1.0 - RANK_FLOOR)
                return _NORMAL.inv_cdf(rank)

    def denormalise(self, value: float) -> float:
        """The raw value a normalised one was made from.

        The inverse of ``normalise`` in arithmetic, not in bits: a value that goes out and comes
        back differs in its last places, so whoever compares the two compares them to a
        tolerance. A robust score at its bound and a quantile score come back as the value at
        the bound and the value interpolated between quantiles, not as the value that went out.
        """
        match self.normalisation:
            case Normalisation.STANDARD:
                return value * self._standard_scale + self.mean
            case Normalisation.ROBUST:
                centre, scale, _ = self._robust_frame
                return value * scale + centre
            case Normalisation.QUANTILE:
                position = _NORMAL.cdf(value) * (len(self.quantiles) - 1)
                below = min(int(position), len(self.quantiles) - 2)
                share = position - below
                return self.quantiles[below] + share * (
                    self.quantiles[below + 1] - self.quantiles[below]
                )

    @property
    def _standard_scale(self) -> float:
        """What a deviation is measured in; one raw unit for a channel that never varied."""
        return self.std if self.std > 0.0 else 1.0

    @property
    def _robust_frame(self) -> tuple[float, float, float]:
        """The median, the robust deviation and the bound; the standard deviation where most agree.

        A channel recorded at one value most of the time has no absolute deviation from its
        median, and measuring its rare departures in raw units would make their size depend on
        the unit the source chose; its standard deviation still says how far the values spread.
        """
        if (
            self.median is None or self.mad is None or self.bound is None
        ):  # pragma: no cover - the constructor's rule
            raise InvalidChannelStatisticsError(
                "only robust statistics have a median, a MAD and a bound"
            )
        robust = MAD_TO_DEVIATION * self.mad
        return self.median, robust if robust > 0.0 else self._standard_scale, self.bound

    def _rank(self, value: float) -> float:
        """The share of training values below ``value``, read off the quantiles.

        A value equal to several knots takes the middle of their ranks, so a channel recorded
        at a few levels spreads them evenly rather than piling each on its lowest rank; between
        two knots the rank is interpolated linearly.
        """
        knots = self.quantiles
        last = len(knots) - 1
        lowest, beyond = bisect_left(knots, value), bisect_right(knots, value)
        if lowest < beyond:
            return (lowest + beyond - 1) / 2 / last
        if lowest == 0:
            return 0.0
        if lowest > last:
            return 1.0
        low, high = knots[lowest - 1], knots[lowest]
        return (lowest - 1 + (value - low) / (high - low)) / last
