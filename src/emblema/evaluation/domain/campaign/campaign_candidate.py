from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod
from emblema.evaluation.domain.campaign.compute_budget import ComputeBudget
from emblema.evaluation.domain.exceptions import (
    CandidateMismatchError,
    InvalidCampaignCandidateError,
)
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class CampaignCandidate:
    """One competitor a campaign names, before anything has been run for it.

    What a candidate is made of is not this context's business beyond the one distinction that
    changes how it is treated: a network is held to the campaign's shared compute budget, a
    classical method reports what it spent. So the budget is stated here for the first kind and
    absent for the second, rather than guessed for both.

    What it starts from and how it is configured are recorded for the same reason a result
    records what it produced: a campaign that named four arms without saying which pretrained
    weights three of them began with, or under which rate and decay, would leave its own
    comparison unidentifiable a month later. The budget says how much arithmetic the arms were
    held to in common; the method says what each of them did with it, and the two are kept apart
    because holding methods to a common value would compare something nobody asked about.

    Invariants: the compute budget is declared exactly when the kind shares one.

    Attributes:
        ref: What the campaign calls this competitor.
        kind: What it is made of, which decides what can run it and how its cost is read.
        budget: What each of its cells may spend learning; ``None`` where the kind shares none.
        method: How it learns within that budget, as its provider states it.
        starts_from: Artifact of the weights it begins with; ``None`` for a candidate that
            begins with none, the control arm included.
    """

    ref: CandidateRef
    kind: CandidateKind
    budget: ComputeBudget | None
    method: CandidateMethod
    starts_from: ArtifactRef | None

    def must_match(self, supplied: Self) -> None:
        """Refuse a candidate supplied under anything other than what this one records.

        Whole rather than field by field: a process set to something the grid never declared
        would answer a point of the curve under it, and every field anyone adds would otherwise
        have to be remembered here again.

        Raises:
            CandidateMismatchError: If the two differ in anything at all.
        """
        if self != supplied:
            raise CandidateMismatchError(
                f"the campaign recorded {self.ref} as something this process does not supply: "
                f"{self} against {supplied}"
            )

    def __post_init__(self) -> None:
        if (self.budget is None) == self.kind.shares_the_compute_budget:
            raise InvalidCampaignCandidateError(
                f"a {self.kind} candidate "
                + (
                    "shares the compute budget and declares none"
                    if self.budget is None
                    else "shares no compute budget and declares one"
                )
            )
