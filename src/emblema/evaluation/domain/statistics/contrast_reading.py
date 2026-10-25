from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.domain.statistics.detection_threshold import DetectionThreshold
from emblema.evaluation.domain.statistics.paired_reading import PairedReading


@dataclass(frozen=True, kw_only=True)
class ContrastReading:
    """A contrast between two recipes at one budget, its threshold, and whether that budget decides.

    Attributes:
        reading: Fine-tuning under the candidate recipe less under the reference.
        threshold: The threshold that difference is read against.
        decides: Whether this budget is one the choice of recipe is made at.
    """

    reading: PairedReading
    threshold: DetectionThreshold
    decides: bool

    @property
    def above(self) -> bool:
        return self.reading.exceeds(self.threshold)

    @property
    def below(self) -> bool:
        return self.reading.falls_below(self.threshold)

    @staticmethod
    def moves(contrasts: Sequence["ContrastReading"]) -> bool:
        """Whether a contrast moves the recipe off its reference.

        It does where it is detected above its threshold at a budget that decides and detected
        below it at none: a reference stays unless a change is shown to help and shown nowhere
        to harm.
        """
        return any(c.decides and c.above for c in contrasts) and not any(c.below for c in contrasts)
