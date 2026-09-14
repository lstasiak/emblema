from dataclasses import dataclass
from enum import Enum

# The confidence every interval of a comparison is stated at.
CONFIDENCE = 0.95


class Verdict(Enum):
    """What an interval of the excess says: above zero, holding zero, or below it.

    ``matched`` is the baseline matching the model, in the plan's words — as trivial as ``beaten``.
    """

    LEARNT = "learnt"
    MATCHED = "matched"
    BEATEN = "beaten"


@dataclass(frozen=True)
class Interval:
    """A bootstrap interval of how much lower the model's error is than a baseline's."""

    low: float
    high: float

    @property
    def verdict(self) -> Verdict:
        if self.low > 0.0:
            return Verdict.LEARNT
        if self.high < 0.0:
            return Verdict.BEATEN
        return Verdict.MATCHED

    def __str__(self) -> str:
        return f"[{self.low:+.4f}, {self.high:+.4f}]"
