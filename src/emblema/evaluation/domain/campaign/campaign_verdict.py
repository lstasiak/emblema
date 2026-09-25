from collections.abc import Sequence
from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.exceptions import (
    InvalidCampaignVerdictError,
    SelectionHasNoVerdictError,
    UnknownCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.task.run_purpose import RunPurpose


@dataclass(frozen=True, kw_only=True)
class CampaignVerdict:
    """What a finished campaign concluded: its one claim, and the shape of the curve around it.

    The endpoint is kept apart from the rest because it is read differently — on its own
    interval, against a reduction promised in advance — while the secondary comparisons are read
    against a correction over the whole family. Mixing them would let a campaign report whichever
    of its cells came out best as though it had been the claim all along. Which side the verdict
    was read on travels with it, because a number read on the validation side is preliminary
    however it came out, and the sentence has to say so.

    Invariants: the verdict was read on a comparing campaign, not a selection; the endpoint is
    not among the secondary comparisons and is not the control's.

    Attributes:
        control: The candidate every comparison is measured against.
        read_on: Which side the campaign ran on, which is what makes its numbers preliminary or
            final.
        endpoint: The single comparison the campaign was designed to test.
        secondary: Every other candidate at every other budget, in reporting order.
    """

    control: CandidateRef
    read_on: RunPurpose
    endpoint: CandidateComparison
    secondary: tuple[CandidateComparison, ...]

    def __post_init__(self) -> None:
        if self.read_on is RunPurpose.SELECTION:
            raise SelectionHasNoVerdictError("a selection chooses a variant and concludes nothing")
        if self.endpoint.candidate == self.control:
            raise InvalidCampaignVerdictError("the endpoint is compared against itself")
        pairings = [(c.candidate, c.budget) for c in self.comparisons()]
        if len(set(pairings)) != len(pairings):
            raise InvalidCampaignVerdictError("a pairing is compared twice")

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
        difference = endpoint.difference
        interval = difference.interval
        found = (
            f"at {_labels(endpoint.budget)} {endpoint.candidate} lowers the error of "
            f"{self.control} by {difference.relative_reduction:.1%} "
            f"({endpoint.control_error.pooled:.3g} → {endpoint.candidate_error.pooled:.3g}, "
            f"a reduction of {difference.reduction:.3g} with {interval.level:.0%} interval "
            f"[{interval.low:.3g}; {interval.high:.3g}], floor {endpoint.floor.value:.3g}"
            f"{self._spread_clause()})"
        )
        if endpoint.verdict is ComparisonVerdict.CONFIRMED:
            opening = f"Confirmed on the registered endpoint: {found}"
        else:
            opening = f"Not confirmed on the registered endpoint ({endpoint.verdict}): {found}"
        return f"{opening}{self._shape_clause()}{self._others_clause()}. {self._side()}"

    def _spread_clause(self) -> str:
        control, candidate = self.endpoint.control_error, self.endpoint.candidate_error
        if control.repeats == 1 and candidate.repeats == 1:
            return "; one repeat each"
        return (
            f"; over {control.repeats} and {candidate.repeats} repeats the sides spread "
            f"{control.spread:.2g} and {candidate.spread:.2g}"
        )

    def _shape_clause(self) -> str:
        along = [c for c in self.secondary if c.candidate == self.endpoint.candidate]
        if not along:
            return ""
        held = [c.budget for c in along if c.verdict is ComparisonVerdict.DISTINGUISHABLE]
        lost = [c.budget for c in along if c.verdict is not ComparisonVerdict.DISTINGUISHABLE]
        if held and lost:
            where = f"at {_spoken(held)}, not at {_spoken(lost)}"
        elif held:
            where = f"at each of them ({_spoken(held)})"
        else:
            where = f"at none of them ({_spoken(lost)})"
        return (
            f"; among the other budgets the advantage of {self.endpoint.candidate} holds under "
            f"the family's correction and above the floor {where}"
        )

    def _others_clause(self) -> str:
        others = [c for c in self.secondary if c.candidate != self.endpoint.candidate]
        if not others:
            return ""
        distinguished = sum(1 for c in others if c.verdict is ComparisonVerdict.DISTINGUISHABLE)
        return (
            f"; {distinguished} of {len(others)} secondary comparisons of the other candidates "
            "are distinguishable"
        )

    def _side(self) -> str:
        if self.read_on is RunPurpose.FINAL:
            return "Test side; final."
        return "Validation side; preliminary."


def _labels(budget: LabelBudget) -> str:
    """A budget read aloud: ``200 labels``, or ``every label`` for the full set."""
    return "every label" if budget.windows is None else f"{budget.windows} labels"


def _spoken(budgets: Sequence[LabelBudget]) -> str:
    """Budgets read aloud as a list: ``50``, ``50 and 200``, ``50, 200 and the full label set``."""
    texts = ["the full label set" if b.windows is None else str(b.windows) for b in budgets]
    if len(texts) == 1:
        return texts[0]
    return f"{', '.join(texts[:-1])} and {texts[-1]}"
