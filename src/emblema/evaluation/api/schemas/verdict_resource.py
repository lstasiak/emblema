from typing import Self

from pydantic import BaseModel, Field

from emblema.evaluation.api.schemas.comparison_resource import ComparisonResource
from emblema.evaluation.application.read_models.verdict_view import VerdictView


class VerdictResource(BaseModel):
    """What a finished campaign concluded: the sentence, and every comparison behind it."""

    sentence: str
    control: str
    read_on: str = Field(description="Which side the numbers were read on: preliminary or final.")
    endpoint: ComparisonResource
    secondary: list[ComparisonResource]

    @classmethod
    def of(cls, verdict: VerdictView) -> Self:
        return cls(
            sentence=verdict.sentence,
            control=str(verdict.control),
            read_on=verdict.read_on.value,
            endpoint=ComparisonResource.of(verdict.endpoint),
            secondary=[ComparisonResource.of(comparison) for comparison in verdict.secondary],
        )
