from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.tuning.tuned_choice import TunedChoice
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository


@dataclass(frozen=True, kw_only=True)
class SelectTunedVariantsCommand:
    """Which selection to read, and which candidate's variants it chose among.

    Attributes:
        campaign: The finished selection campaign.
        candidate: The candidate whose variants it ran.
    """

    campaign: CampaignId
    candidate: CandidateRef


class SelectTunedVariants:
    """Reads what a finished selection chose for a candidate, at every budget it ran.

    Nothing is stored: the choices are what a comparison's file then names, and naming them
    there is what dates them. The same rule reads them again when that comparison is declared,
    so this is the question asked early, not an answer anyone has to take on trust.
    """

    def __init__(self, campaigns: EvaluationCampaignRepository) -> None:
        self._campaigns = campaigns

    def __call__(self, command: SelectTunedVariantsCommand) -> tuple[TunedChoice, ...]:
        """One choice per budget of the selection, in the order it ran its budgets.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            SelectionNotReadableError: If it is not a finished selection, or holds no choice
                for the candidate.
        """
        selection = self._campaigns.get(command.campaign)
        return tuple(
            TunedChoice(
                candidate=command.candidate,
                budget=budget,
                variant=selection.selected(command.candidate, budget),
                selected_by=selection.campaign_id,
            )
            for budget in selection.design.budgets
        )
