from dataclasses import dataclass
from typing import Final

# How a channel's value was normalised: the vocabulary the Catalog's normalisation is published
# under. The standard score is the one every manifest written before the others existed uses.
NORMALISATION_STANDARD = "standard"
NORMALISATION_ROBUST = "robust"
NORMALISATION_QUANTILE = "quantile"
NORMALISATIONS = (NORMALISATION_STANDARD, NORMALISATION_ROBUST, NORMALISATION_QUANTILE)
# The bound every robust score was held to before a manifest could state one, so a robust
# manifest that states none is read at it and one written at it keeps its bytes.
ROBUST_BOUND_UNSTATED: Final = 10.0


@dataclass(frozen=True)
class PublishedChannelStatistics:
    """Parameters the training units established for one channel's normalisation.

    Attributes:
        count: Values the statistics were computed over.
        mean: Arithmetic mean.
        std: Population standard deviation.
        normalisation: One of ``NORMALISATIONS``: how a value was turned into the token's value.
        median: Median, where the normalisation is robust.
        mad: Median absolute deviation from the median, where the normalisation is robust.
        quantiles: Values at evenly spaced ranks from the lowest to the highest, where the
            normalisation is by quantile.
        bound: How far from the median a robust score may lie, in robust deviations, where the
            normalisation is robust.
    """

    count: int
    mean: float
    std: float
    normalisation: str = NORMALISATION_STANDARD
    median: float | None = None
    mad: float | None = None
    quantiles: tuple[float, ...] = ()
    bound: float | None = None
