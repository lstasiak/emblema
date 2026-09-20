from enum import Enum


class ComparisonVerdict(Enum):
    """What the registered rules say about one cell compared with the control arm.

    Attributes:
        CONFIRMED: The endpoint's claim holds: the registered share of the control's error is
            taken off, the whole interval lies above zero and the reduction clears the floor.
        DISTINGUISHABLE: A secondary cell whose advantage survives the family's correction and
            clears the floor; it describes the curve's shape and confirms nothing on its own.
        BELOW_REGISTERED_REDUCTION: The endpoint's reduction is distinguishable and clears the
            floor but falls short of the registered share.
        PRACTICALLY_NIL: Distinguishable, but smaller than the practical floor.
        INDISTINGUISHABLE: Zero is not excluded — by the interval for the endpoint, by the
            family's correction for a secondary cell.
        WORSE: Distinguishable the wrong way: the candidate's error is higher.
    """

    CONFIRMED = "confirmed"
    DISTINGUISHABLE = "distinguishable"
    BELOW_REGISTERED_REDUCTION = "distinguishable, below the registered reduction"
    PRACTICALLY_NIL = "distinguishable, practically nil"
    INDISTINGUISHABLE = "indistinguishable"
    WORSE = "worse"

    def __str__(self) -> str:
        return self.value
