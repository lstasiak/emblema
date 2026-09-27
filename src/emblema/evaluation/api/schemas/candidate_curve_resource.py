"""A candidate's curve as the API shows it: the curve, and each of its points."""

from typing import Self

from pydantic import BaseModel, Field

from emblema.evaluation.application.read_models.candidate_curve import CandidateCurve


class CurvePointResource(BaseModel):
    """What a candidate scored at one budget, pooled over the repeats that have run."""

    budget: str
    error: float | None = Field(description="Pooled over the repeats; absent while none has run.")
    spread: float
    repeats: int


class CandidateCurveResource(BaseModel):
    """One candidate's error over the budgets, in reporting order."""

    candidate: str
    kind: str
    points: list[CurvePointResource]

    @classmethod
    def of(cls, curve: CandidateCurve) -> Self:
        return cls(
            candidate=str(curve.candidate),
            kind=curve.kind.value,
            points=[
                CurvePointResource(
                    budget=point.budget,
                    error=point.error,
                    spread=point.spread,
                    repeats=point.repeats,
                )
                for point in curve.points
            ],
        )
