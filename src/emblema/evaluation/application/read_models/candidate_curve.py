"""The label-efficiency curve of one candidate: its error at every budget the campaign ran."""

from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats


@dataclass(frozen=True, kw_only=True)
class CurvePoint:
    """What a candidate scored at one budget, pooled over the repeats that have run.

    Attributes:
        budget: How many labelled windows the candidate learnt from, as text.
        error: The error under the campaign's measure, pooled over every repeat; ``None`` while
            no repeat has run.
        spread: The standard deviation of the error over the repeats; zero for one.
        repeats: How many repeats have run.
    """

    budget: str
    error: float | None
    spread: float
    repeats: int

    @classmethod
    def of(cls, reading: CampaignReading, candidate: CandidateRef, budget: LabelBudget) -> Self:
        results = reading.results_of(candidate, budget)
        if not results:
            return cls(budget=budget.text(), error=None, spread=0.0, repeats=0)
        error = ErrorOverRepeats.of(
            reading.error_of(candidate, budget),
            [result.error_under(reading.campaign.design.measure) for result in results],
        )
        return cls(
            budget=budget.text(), error=error.pooled, spread=error.spread, repeats=error.repeats
        )


@dataclass(frozen=True, kw_only=True)
class CandidateCurve:
    """One candidate's error over the budgets, in the order the campaign reports them.

    Attributes:
        candidate: What the campaign calls the competitor.
        kind: What it is made of.
        points: One point per budget of the grid.
    """

    candidate: CandidateRef
    kind: CandidateKind
    points: tuple[CurvePoint, ...]

    @classmethod
    def of(cls, reading: CampaignReading, candidate: CampaignCandidate) -> Self:
        return cls(
            candidate=candidate.ref,
            kind=candidate.kind,
            points=tuple(
                CurvePoint.of(reading, candidate.ref, budget)
                for budget in reading.campaign.design.budgets
            ),
        )
