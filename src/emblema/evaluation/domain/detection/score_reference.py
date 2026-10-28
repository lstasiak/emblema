from enum import StrEnum


class ScoreReference(StrEnum):
    """What a detector's distance is read against before the alarm is applied.

    A detector fitted once on the fitting side reads every later window against that one
    standard, so a recording that drifts away from it raises alarms everywhere and the events
    are lost among them. Reading each window against the other windows of its own unit instead
    asks how unusual it is for that month or that experiment; no label is spent, since the unit's
    windows arrive unlabelled, and a window alone in its unit has nothing to be read against.

    Attributes:
        FITTING_SIDE: The distance itself, every window held to the standard the fit set.
        OWN_UNIT: The distance less the median of its unit's distances, over their spread
            (the median absolute deviation, scaled to a standard deviation), the fitting side's
            windows read the same way within their own units when the alarm is set.
    """

    FITTING_SIDE = "fitting"
    OWN_UNIT = "unit"
