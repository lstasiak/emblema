from typing import Self

from pydantic import BaseModel, Field

from emblema.evaluation.application.read_models.comparison_view import ComparisonView


class ComparisonResource(BaseModel):
    """One candidate against the control at one budget, every figure the verdict rests on."""

    candidate: str
    budget: str
    control_error: float
    control_spread: float
    control_repeats: int
    candidate_error: float
    candidate_spread: float
    candidate_repeats: int
    reduction: float
    relative_reduction: float | None = Field(
        description="The reduction as a share of the control's error; null where it made none."
    )
    interval_low: float
    interval_high: float
    interval_level: float
    p_value: float
    floor: float
    verdict: str

    @classmethod
    def of(cls, comparison: ComparisonView) -> Self:
        return cls(
            candidate=str(comparison.candidate),
            budget=comparison.budget,
            control_error=comparison.control_error,
            control_spread=comparison.control_spread,
            control_repeats=comparison.control_repeats,
            candidate_error=comparison.candidate_error,
            candidate_spread=comparison.candidate_spread,
            candidate_repeats=comparison.candidate_repeats,
            reduction=comparison.reduction,
            relative_reduction=comparison.relative_reduction,
            interval_low=comparison.interval_low,
            interval_high=comparison.interval_high,
            interval_level=comparison.interval_level,
            p_value=comparison.p_value,
            floor=comparison.floor,
            verdict=comparison.verdict,
        )
