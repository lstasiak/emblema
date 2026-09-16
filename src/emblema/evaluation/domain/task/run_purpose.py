from enum import Enum


class RunPurpose(Enum):
    """What a run is for, which decides whether the frozen test side may be opened for it.

    The distinction is the whole reason a test set stays untouched: a project that tunes on its
    test data reports a number that cannot be checked by anyone, including itself.

    Attributes:
        TUNING: A run whose outcome informs a decision — a hyperparameter, a variant, whether to
            carry on. It sees the tuning and validation sides and never the test side.
        FINAL: The one run per task whose numbers are published, made once every decision has
            already been taken.
    """

    TUNING = "tuning"
    FINAL = "final"
