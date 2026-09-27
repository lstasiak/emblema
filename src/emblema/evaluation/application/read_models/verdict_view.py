from dataclasses import dataclass
from typing import Self

from emblema.evaluation.application.read_models.comparison_view import ComparisonView
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.task.run_purpose import RunPurpose


@dataclass(frozen=True, kw_only=True)
class VerdictView:
    """What a finished campaign concluded, as a client shows it: the sentence and every figure.

    Attributes:
        sentence: The verdict in one sentence.
        control: The candidate every comparison is measured against.
        read_on: Which side the campaign ran on, which makes its numbers preliminary or final.
        endpoint: The single comparison the campaign was designed to test.
        secondary: Every other comparison, in reporting order.
    """

    sentence: str
    control: CandidateRef
    read_on: RunPurpose
    endpoint: ComparisonView
    secondary: tuple[ComparisonView, ...]

    @classmethod
    def of(cls, verdict: CampaignVerdict) -> Self:
        return cls(
            sentence=verdict.sentence(),
            control=verdict.control,
            read_on=verdict.read_on,
            endpoint=ComparisonView.of(verdict.endpoint),
            secondary=tuple(ComparisonView.of(comparison) for comparison in verdict.secondary),
        )
