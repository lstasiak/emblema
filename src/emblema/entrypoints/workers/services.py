from dataclasses import dataclass

from emblema.evaluation.application.use_cases.advance_campaign import AdvanceCampaign
from emblema.evaluation.application.use_cases.complete_campaign import CompleteCampaign
from emblema.evaluation.application.use_cases.define_campaign import DefineCampaign
from emblema.evaluation.application.use_cases.open_test_split import OpenTestSplit
from emblema.evaluation.application.use_cases.run_campaign_cell import RunCampaignCell


@dataclass(frozen=True)
class Services:
    """The use cases this process can run, each already holding its dependencies.

    The worker both runs cells and submits them, because the two are how a campaign starts and
    how it recovers: a process that could only consume would need a second one to resume a grid
    after a broker was restarted.
    """

    define_campaign: DefineCampaign
    advance_campaign: AdvanceCampaign
    run_campaign_cell: RunCampaignCell
    complete_campaign: CompleteCampaign
    open_test_split: OpenTestSplit
