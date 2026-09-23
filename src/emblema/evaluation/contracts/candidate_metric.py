from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.contracts.exceptions import InvalidCandidateMetricError


@dataclass(frozen=True, kw_only=True)
class CandidateMetric:
    """What a candidate scored at one operating point of a campaign.

    The budget is a count of labelled windows rather than this context's value object, and the
    metric is named rather than assumed: a consumer reads the number and what it is without
    knowing how this context spells either, which is the whole point of publishing it.

    Invariants: the metric is non-blank without surrounding whitespace; a counted budget asks
    for at least one window; the value is finite; at least one repeat stands behind it.

    Attributes:
        metric: What was measured, as the campaign names it.
        budget: How many labelled windows the candidate learnt from, or ``None`` for every window
            the task's tuning side holds.
        value: The figure, pooled over the repeats.
        repeats: How many runs it was pooled over, so a reader knows how much stands behind it.
    """

    metric: str
    budget: int | None
    value: float
    repeats: int

    def __post_init__(self) -> None:
        if not self.metric or self.metric != self.metric.strip():
            raise InvalidCandidateMetricError(
                "a metric must be named, non-blank without surrounding whitespace"
            )
        if self.budget is not None and self.budget < 1:
            raise InvalidCandidateMetricError(
                f"a counted budget asks for at least one window, got {self.budget}"
            )
        if not isfinite(self.value):
            raise InvalidCandidateMetricError(f"a metric must be finite, got {self.value}")
        if self.repeats < 1:
            raise InvalidCandidateMetricError(
                f"a metric stands on at least one repeat, got {self.repeats}"
            )
