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

    Attributes:
        NEAREST_NEIGHBOURS: The mean Euclidean distance to the nearest fitting windows.
        WHITENED: The squared Mahalanobis distance from the fitting mean under the fitting
            side's covariance, shrunk so it inverts whatever the side's size.
    """

    NEAREST_NEIGHBOURS = "neighbours"
    WHITENED = "whitened"
