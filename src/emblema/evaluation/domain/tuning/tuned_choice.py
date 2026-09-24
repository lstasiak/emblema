from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.exceptions import InvalidTunedChoiceError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.tuning.candidate_variant import CandidateVariant


@dataclass(frozen=True, kw_only=True)
class TunedChoice:
    """Which variant of a candidate a comparison runs at one budget, and which selection chose it.

    Per budget, because the knobs that suit fifty labels are not the ones that suit a thousand,
    and a candidate held at the setting chosen for one point of a curve is mistuned at every
    other. The selection is named so the choice can be checked against it rather than trusted.

    Invariants: the variant is one of the candidate's.

    Attributes:
        candidate: The candidate as the comparison names it.
        budget: The budget the variant runs at.
        variant: The variant the selection chose.
        selected_by: The finished selection campaign that chose it.
    """

    candidate: CandidateRef
    budget: LabelBudget
    variant: CandidateRef
    selected_by: CampaignId

    def __post_init__(self) -> None:
        if CandidateVariant.parse(self.variant).base != self.candidate:
            raise InvalidTunedChoiceError(f"{self.variant} is not a variant of {self.candidate}")
