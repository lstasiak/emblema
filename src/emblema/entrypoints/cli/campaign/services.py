from dataclasses import dataclass

from emblema.evaluation.application.use_cases.accept_campaign_order_result import (
    AcceptCampaignOrderResult,
)
from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.announce_campaign import AnnounceCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask
from emblema.evaluation.application.use_cases.order_campaign_cells import OrderCampaignCells
from emblema.evaluation.application.use_cases.select_tuned_variants import SelectTunedVariants


@dataclass(frozen=True)
class Services:
    """The use cases this command line can run: define, declare, submit, order out, accept back.

    Announcing a closed campaign again is among them because delivering its conclusion is this
    process's business as much as a worker's: whichever closed it published in process, and a
    delivery that failed there is repaired by whoever holds the registry.

    Running a cell is not among them, and deliberately: a grid is worked by the processes that
    consume the queues, or by the process that runs an order, and both answer through the same
    candidates. A command line that ran one here would be a third way of producing the same
    numbers, under whatever the machine at the keyboard happened to be configured with.
    """

    define_downstream_task: DefineDownstreamTask
    define_campaign: DefineCampaign
    advance_campaign: AdvanceCampaign
    announce_campaign: AnnounceCampaign
    order_campaign_cells: OrderCampaignCells
    accept_campaign_order_result: AcceptCampaignOrderResult
    select_tuned_variants: SelectTunedVariants
