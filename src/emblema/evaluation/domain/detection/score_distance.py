from enum import StrEnum


class ScoreDistance(StrEnum):
    """How far a window's features lie from the normal ones: the distance a detector scores by.

    Every feature is standardised over the fitting side either way. A distance to the nearest
    fitting windows then weighs the leading directions of the fitting variation most, since the
    features are correlated and those directions carry most of their spread; a change spread
    thinly over many directions barely moves it. A whitened distance weighs every direction of
    the fitting variation alike, so it reads such a change at its full size, and it reads a
    month that drifted along a leading direction as less unusual than the nearest-neighbour
    distance does.

    Three more are the bars the field holds a representation to: the plain squared distance from
    the fitting mean, which weighs every feature's departure alike; the farthest single feature's
    departure, which flags a window for one reading far out, as a threshold on a channel's
    deviations would; and the classical reading of principal components, how far a window lies
    within the subspace the leading components span and how far outside it.

    Attributes:
        NEAREST_NEIGHBOURS: The mean Euclidean distance to the nearest fitting windows.
        WHITENED: The squared Mahalanobis distance from the fitting mean under the fitting
            side's covariance, shrunk so it inverts whatever the side's size.
        CENTRE: The squared Euclidean distance from the fitting mean, features standardised.
        FARTHEST: The largest standardised departure of any one feature from the fitting mean.
        SUBSPACE: Hotelling's statistic within the leading components' subspace plus the squared
            residual outside it, each over its fitting median.
    """

    NEAREST_NEIGHBOURS = "neighbours"
    WHITENED = "whitened"
    CENTRE = "centre"
    FARTHEST = "farthest"
    SUBSPACE = "subspace"
