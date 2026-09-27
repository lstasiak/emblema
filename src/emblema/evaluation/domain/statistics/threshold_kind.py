from enum import StrEnum


class ThresholdKind(StrEnum):
    """How a registered threshold on a reduction is stated: as a share, or in the error's unit.

    A share of the control's error is scale-free, which suits an error with no natural bound:
    a tenth of the remaining life missed reads the same on any turbofan. An error bounded by a
    perfect answer is different — a share of one minus an area under the ROC curve is a share of
    the distance left to perfect ranking, so the same share asks twice as much of a candidate
    against a weak control as against a strong one. Where the literature states its differences
    in the measure's own unit, a registration that wants to be read beside it does too.

    Attributes:
        RELATIVE: A share of the control's error.
        ABSOLUTE: An amount in the error's own unit.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"
