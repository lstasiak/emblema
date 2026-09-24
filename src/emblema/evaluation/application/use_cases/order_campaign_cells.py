from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.jobs.worker_pool import WorkerPool
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class OrderCampaignCellsCommand:
    """Which campaign's outstanding cells to hand out, of which pool, at which revision of code.

    Attributes:
        campaign: The campaign whose cells are ordered.
        pool: Which kind of process will run them; only the cells that kind can run go out.
        git_commit: Revision of the code the cells are to be run with.
    """

    campaign: CampaignId
    pool: WorkerPool
    git_commit: str


class OrderCampaignCells:
    """Writes a campaign's outstanding cells of one pool into an order for another machine.

    The counterpart of handing the cells to a queue, for a machine the queue cannot reach. The
    cells go out as the same requests a worker would put to its candidates, built by the
    campaign itself, so a cell answered through an order and one answered from the queue are
    the same question. Only cells without a result go out: ordering again after a result was
    accepted hands out what is still missing and nothing twice.
    """

    def __init__(
        self,
        campaigns: EvaluationCampaignRepository,
        tasks: DownstreamTaskRepository,
        handoff: CampaignHandoff,
    ) -> None:
        self._campaigns = campaigns
        self._tasks = tasks
        self._handoff = handoff

    def __call__(self, command: OrderCampaignCellsCommand) -> ArtifactRef:
        """Place the order and return the reference the other machine is handed.

        Raises:
            CampaignNotFoundError: If the campaign is unknown.
            TaskNotFoundError: If the campaign's task is unknown.
            InvalidCampaignOrderError: If no outstanding cell belongs to that pool, or the
                commit is blank.
        """
        campaign = self._campaigns.get(command.campaign)
        needs_the_stack = command.pool is WorkerPool.ML
        return self._handoff.place(
            CampaignOrder(
                campaign=campaign.campaign_id,
                task=self._tasks.get(campaign.task),
                evaluations=tuple(
                    campaign.evaluation_of(cell)
                    for cell in campaign.pending()
                    if campaign.design.get_candidate(cell.candidate).kind.needs_the_ml_stack
                    == needs_the_stack
                ),
                git_commit=command.git_commit,
            )
        )
