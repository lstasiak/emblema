from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.exceptions import UnknownCandidateError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict


@dataclass(frozen=True, kw_only=True)
class CampaignVerdict:
    """What a finished campaign concluded: its one claim, and the shape of the curve around it.

    The endpoint is kept apart from the rest because it is read differently — on its own
    interval, against a reduction promised in advance — while the secondary comparisons are read
    against a correction over the whole family. Mixing them would let a campaign report whichever
    of its cells came out best as though it had been the claim all along.

    Attributes:
        endpoint: The single comparison the campaign was designed to test.
        secondary: Every other candidate at every other budget, in reporting order.
    """

    endpoint: CandidateComparison
    secondary: tuple[CandidateComparison, ...]

    def comparisons(self) -> tuple[CandidateComparison, ...]:
        """Every comparison the campaign made, the endpoint first."""
        return (self.endpoint, *self.secondary)

    def get_comparison(self, candidate: CandidateRef, budget: LabelBudget) -> CandidateComparison:
        """What the campaign made of this candidate at this budget.

        Raises:
            UnknownCandidateError: If the campaign compared no such pairing.
        """
        for comparison in self.comparisons():
            if comparison.candidate == candidate and comparison.budget == budget:
                return comparison
        raise UnknownCandidateError(f"no comparison of {candidate} at a budget of {budget.text()}")

    def sentence(self) -> str:
        """The verdict in one sentence, for whoever reads the outcome rather than the grid."""
        endpoint = self.endpoint
        interval = endpoint.difference.interval
        distinguished = sum(
            1
            for comparison in self.secondary
            if comparison.verdict is ComparisonVerdict.DISTINGUISHABLE
        )
        return (
            f"At {endpoint.budget.text()} labels {endpoint.candidate} against the control is "
            f"{endpoint.verdict}: a reduction of {endpoint.difference.reduction:.3g} "
            f"({endpoint.difference.relative_reduction:.1%}), interval "
            f"[{interval.low:.3g}; {interval.high:.3g}] at {interval.level:.0%}, "
            f"practical floor {endpoint.floor.value:.3g}; "
            f"{distinguished} of {len(self.secondary)} secondary comparisons are distinguishable."
        )
