from dataclasses import dataclass

from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.announce_campaign import AnnounceCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.define_downstream_task import DefineDownstreamTask


@dataclass(frozen=True)
class Services:
    """The use cases this command line can run: define a task, declare a campaign, submit it.

    Announcing a closed campaign again is among them because delivering its conclusion is this
    process's business as much as a worker's: whichever closed it published in process, and a
    delivery that failed there is repaired by whoever holds the registry.

    Running a cell is not among them, and deliberately: a grid is worked by the processes that
    consume the queues, so a command line that ran one would be a second way of producing the
    same numbers, under whatever the machine at the keyboard happened to be configured with.
    """

    define_downstream_task: DefineDownstreamTask
    define_campaign: DefineCampaign
    advance_campaign: AdvanceCampaign
    announce_campaign: AnnounceCampaign
