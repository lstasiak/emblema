from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.jobs.queued_job import QueuedJob
from emblema.shared.jobs.worker_pool import WorkerPool
from emblema.shared.ports.job_queue import JobQueue

RUN_CAMPAIGN_CELL = "evaluation.run_campaign_cell"
"""What a worker knows one cell of a campaign as."""


@dataclass(frozen=True, kw_only=True)
class AdvanceCampaignCommand:
    """Request to hand a campaign's outstanding work to the queue.

    Attributes:
        campaign: Which campaign to advance.
    """

    campaign: CampaignId


class AdvanceCampaign:
    """Submits every cell of a campaign that has not run, and says how many that was.

    This is both how a campaign starts and how it recovers. A campaign that has run nothing has
    its whole grid outstanding; one that lost a worker mid-grid has whatever was not recorded.
    The two are the same operation because the campaign is what knows how far it got, so
    resuming is submitting the same work again rather than reconstructing a position.

    Submitting a cell twice is harmless: a worker that receives a cell already recorded reports
    what is stored instead of running it again. That is what lets this be called after a crash
    without first working out which messages survived.
    """

    def __init__(self, campaigns: EvaluationCampaignRepository, jobs: JobQueue) -> None:
        self._campaigns = campaigns
        self._jobs = jobs

    def __call__(self, command: AdvanceCampaignCommand) -> int:
        """Submit the campaign's pending cells; return how many were handed over.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            JobQueueError: If the queue cannot be reached; cells already submitted stay
                submitted, and calling again re-submits the rest.
        """
        campaign = self._campaigns.get(command.campaign)
        pending = campaign.pending()
        for cell in pending:
            self._jobs.submit(self._job(campaign, cell))
        return len(pending)

    @staticmethod
    def _job(campaign: EvaluationCampaign, cell: CampaignCell) -> QueuedJob:
        kind = campaign.design.get_candidate(cell.candidate).kind
        return QueuedJob(
            name=RUN_CAMPAIGN_CELL,
            pool=WorkerPool.ML if kind.shares_the_compute_budget else WorkerPool.GENERAL,
            arguments={
                "campaign": str(campaign.campaign_id),
                "candidate": str(cell.candidate),
                "budget": cell.budget.text(),
                "seed": cell.seed,
            },
        )
