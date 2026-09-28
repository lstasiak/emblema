from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison


@dataclass(frozen=True, kw_only=True)
class ComparisonView:
    """One candidate against the control at one budget, every figure the verdict rests on, flat.

    Attributes:
        candidate: Which competitor was compared.
        budget: How many labelled windows both sides learnt from, as text.
        control_error: What the control scored, pooled over its repeats.
        control_spread: How the control's repeats spread.
        control_repeats: How many repeats the control ran.
        candidate_error: What the candidate scored, pooled over its repeats.
        candidate_spread: How the candidate's repeats spread.
        candidate_repeats: How many repeats the candidate ran.
        reduction: How much lower the candidate's error is than the control's.
        relative_reduction: The reduction as a share of the control's error; ``None`` where the
            control made no error.
        interval_low: Lower end of the interval the reduction lies in.
        interval_high: Upper end of that interval.
        interval_level: The confidence the interval is stated at, as a share.
        p_value: Two-sided, over resamples of the units.
        floor: The smallest reduction that counts as a difference at this budget.
        verdict: What the campaign's rules made of the three, in words.
    """

    candidate: CandidateRef
    budget: str
    control_error: float
    control_spread: float
    control_repeats: int
    candidate_error: float
    candidate_spread: float
    candidate_repeats: int
    reduction: float
    relative_reduction: float | None
    interval_low: float
    interval_high: float
    interval_level: float
    p_value: float
    floor: float
    verdict: str

    @classmethod
    def of(cls, comparison: CandidateComparison) -> Self:
        difference = comparison.difference
        return cls(
            candidate=comparison.candidate,
            budget=comparison.budget.text(),
            control_error=comparison.control_error.pooled,
            control_spread=comparison.control_error.spread,
            control_repeats=comparison.control_error.repeats,
            candidate_error=comparison.candidate_error.pooled,
            candidate_spread=comparison.candidate_error.spread,
            candidate_repeats=comparison.candidate_error.repeats,
            reduction=difference.reduction,
            relative_reduction=difference.relative_reduction,
            interval_low=difference.interval.low,
            interval_high=difference.interval.high,
            interval_level=difference.interval.level,
            p_value=difference.p_value,
            floor=comparison.floor.value,
            verdict=str(comparison.verdict),
        )
