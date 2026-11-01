from enum import StrEnum


class TargetKind(StrEnum):
    """What kind of number a task's label is, and so how a candidate is taught and judged on it.

    A property of the label scheme rather than of the candidate: the loss a network descends,
    the value its head starts from, the transform its answer passes through and the error a
    comparison reads are all fixed by what the target is, so every candidate of a campaign
    learns and is scored the same way whatever it is built from.

    Attributes:
        CONTINUOUS: A quantity on a scale, answered in the task's own unit: squared error is
            what the candidate descends and what the comparison reads.
        BINARY: One of two outcomes, answered as the probability of the positive one: the
            candidate descends the negative log-likelihood of that probability, and the
            comparison reads how well the answers rank the outcomes apart.
        ANOMALY: Which event, if any, a window overlaps, answered as a score that rises with
            how unlike the normal data the window is: no candidate learns it per window, the
            fitting side carrying no label beyond the few events a detector is taught, and the
            comparison reads how well the scores rank the events' windows above the normal ones.
    """

    CONTINUOUS = "continuous"
    BINARY = "binary"
    ANOMALY = "anomaly"
