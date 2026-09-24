from dataclasses import dataclass
from math import isfinite

from emblema.serving.domain.exceptions import InvalidCampaignScoreError


@dataclass(frozen=True, kw_only=True)
class CampaignScore:
    """What a campaign reported a candidate to have scored at one operating point.

    Kept so that whoever promotes can see what they are choosing between, and never acted on:
    ranking candidates is the verdict's business, and a ranking made here would be a second
    verdict under rules nobody registered.

    Invariants: the metric is non-blank without surrounding whitespace; a counted budget asks
    for at least one window; the value is finite; at least one repeat stands behind it.

    Attributes:
        metric: What was measured, as the campaign named it.
        budget: How many labelled windows the candidate learnt from, or ``None`` for every
            window its task's tuning side holds.
        value: The figure, pooled over the repeats.
        repeats: How many runs it was pooled over.
    """

    metric: str
    budget: int | None
    value: float
    repeats: int

    def __post_init__(self) -> None:
        if not self.metric or self.metric != self.metric.strip():
            raise InvalidCampaignScoreError(
                "a score names its metric, non-blank without surrounding whitespace"
            )
        if self.budget is not None and self.budget < 1:
            raise InvalidCampaignScoreError(
                f"a counted budget asks for at least one window, got {self.budget}"
            )
        if not isfinite(self.value):
            raise InvalidCampaignScoreError(f"a score must be finite, got {self.value}")
        if self.repeats < 1:
            raise InvalidCampaignScoreError(
                f"a score stands on at least one repeat, got {self.repeats}"
            )
