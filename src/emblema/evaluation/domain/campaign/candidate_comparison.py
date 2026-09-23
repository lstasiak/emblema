from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor


@dataclass(frozen=True, kw_only=True)
class CandidateComparison:
    """One candidate against the control at one budget, with what the registered rules made of it.

    The difference, the floor it was held to and the verdict travel together because none of
    them is readable alone: a reduction without its interval says nothing about whether it is
    real, and a verdict without the floor it cleared says nothing about whether it matters.

    Attributes:
        candidate: Which competitor was compared.
        budget: How many labelled windows both sides learnt from.
        difference: The reduction over the control, with its interval and p-value.
        floor: The smallest reduction that counts as a difference at this budget.
        verdict: What the campaign's rules made of the three.
    """

    candidate: CandidateRef
    budget: LabelBudget
    difference: PairedDifference
    floor: PracticalFloor
    verdict: ComparisonVerdict
