from enum import StrEnum


class TaughtHead(StrEnum):
    """How a detector taught a few of the fitting side's events scores a window by them.

    A detector taught events replaces the distance of its zero-label kin with a rule the events
    set: the field's linear head over the same features, or the field's comparator, trees told
    the taught windows apart from the rest of the fitting side with the same labels. The heads
    are fixed by rule, with nothing chosen on the scored side.

    Attributes:
        DIRECTION: Fisher's direction under the fitting side's shrunk covariance: a window's
            whitened coordinate along the line from the fitting mean to the taught events' mean,
            signed, so a window on the events' side scores high and one on the far side low.
        COMBINED: The mean of two robustly standardised scores, the squared whitened distance
            from the fitting mean and the direction, so an event unlike the taught ones still
            raises the first while one like them raises both; a product candidate, read
            descriptively.
        TREES: Gradient-boosted trees over the same features, the taught windows against the
            rest of the fitting side, scored by their log-odds; the classical comparator with
            the same labels.
    """

    DIRECTION = "direction"
    COMBINED = "combined"
    TREES = "trees"
