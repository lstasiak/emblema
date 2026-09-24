from dataclasses import dataclass

from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.events.domain_event import EventId
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True, kw_only=True)
class CompleteCampaignCommand:
    """Request to close a campaign whose grid is whole and read its verdict.

    Attributes:
        campaign: Which campaign to close.
    """

    campaign: CampaignId


class CompleteCampaign:
    """Closes a finished grid, reads the verdict its rules give, and publishes it.

    The verdict is read off the campaign as closed, because closing is what makes reading it
    legitimate: a grid read while cells are outstanding reports the ones that happened to finish,
    and which those are is never independent of what they found. It is read before the closed
    campaign is stored, so a verdict that cannot be read leaves the campaign open to be closed
    again rather than closed for good with nothing announced; and the campaign is stored before
    the message goes out, so nothing is announced that was not first recorded.

    A selection closes and announces nothing. What it answers is which variant a comparison
    should run, read from its results when the comparison is declared; it has no verdict and no
    candidate for another context to promote.
    """

    def __init__(
        self,
        campaigns: EvaluationCampaignRepository,
        outcomes: CampaignCompletedAssembler,
        clock: Clock,
        ids: IdGenerator,
        events: EventPublisher,
    ) -> None:
        self._campaigns = campaigns
        self._outcomes = outcomes
        self._clock = clock
        self._ids = ids
        self._events = events

    def __call__(self, command: CompleteCampaignCommand) -> CampaignCompleted | None:
        """Close the campaign, publish what it concluded, and return the message.

        A selection is closed and ``None`` returned, since it concludes nothing to publish.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            IncompleteCampaignError: If any cell of its grid has not run.
            CampaignClosedError: If it has already been closed.
            CampaignChangedElsewhereError: If another process closed it first, which is what
                keeps a grid from announcing its own conclusion twice.
            InvalidPairedUnitErrorsError: If a candidate and the control were scored on
                different units.
        """
        at = self._clock.now()
        read = self._campaigns.get(command.campaign)
        campaign = read.complete(at)
        if campaign.selects:
            self._campaigns.save(campaign, seen=read.revision)
            return None
        completed = self._outcomes.assemble(
            campaign,
            campaign.verdict(),
            event_id=self._ids.generate(EventId),
            occurred_at=at,
        )
        self._campaigns.save(campaign, seen=read.revision)
        self._events.publish(completed)
        return completed
