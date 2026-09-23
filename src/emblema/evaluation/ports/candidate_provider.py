from typing import Protocol

from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_candidate import CampaignCandidate
from emblema.evaluation.domain.campaign.candidate_evaluation import CandidateEvaluation
from emblema.evaluation.domain.campaign.cell_result import CellResult


class CandidateProvider(Protocol):
    """Supplies a campaign's competitors: what each one is, and what it answers with.

    This is the whole of what this context knows about the things it compares. A network
    adapted from pretrained weights and a classical method fitted from scratch reach a campaign
    through the same two calls, which is what lets a campaign be made of either without knowing
    which — and what keeps this context from depending on the one that trains backbones.

    Nothing about how a candidate is built crosses the port: no weights, no checkpoint, no
    hyperparameter. A candidate is named, and what that name is made of belongs to whoever
    supplies it.
    """

    def describe(self, candidate: CandidateRef) -> CampaignCandidate:
        """What this candidate is, in the terms a design is stated in.

        Asked while a campaign is being designed, so that the budget a candidate is held to is
        the one it will actually spend rather than one written down beside it.

        Raises:
            UnknownCandidateError: If this provider supplies no such candidate.
        """
        ...

    def evaluate(self, request: CandidateEvaluation) -> CellResult:
        """Run one cell and report what the candidate got wrong, unit by unit.

        Raises:
            UnknownCandidateError: If this provider supplies no such candidate.
            TaskNotFoundError: If the task is unknown.
            FrozenTestSplitClosedError: If a run that is not the final one asked for the frozen
                side.
            InvalidLabelBudgetError: If the tuning side holds fewer windows than asked for.
        """
        ...
