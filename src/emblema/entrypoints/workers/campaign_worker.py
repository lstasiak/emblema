from collections.abc import Mapping

from emblema.evaluation.application.use_cases.run_campaign_cell import (
    RunCampaignCell,
    RunCampaignCellCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId, CandidateRef
from emblema.evaluation.domain.campaign.campaign_cell import CampaignCell
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.jobs.job_argument import JobArgument


class CampaignWorker:
    """What a worker process does with a job, apart from the queue that delivered it.

    The same for whichever pool the process serves: reading a message back into a command is
    what an entry point is for, and which candidates answer it was settled when the process was
    assembled. No decision is made here — a cell already recorded is recognised by the use case,
    not by this.

    It holds the one use case rather than the whole composition, so that what a job can reach is
    what a job needs, and so that a test of reading a message back needs nothing built.
    """

    def __init__(self, run_campaign_cell: RunCampaignCell) -> None:
        self._run = run_campaign_cell

    def run_campaign_cell(self, arguments: Mapping[str, JobArgument]) -> None:
        """Run the cell a job names.

        Raises:
            ValueError: If the job does not name a cell.
            InvalidLabelBudgetError: If the budget it names is not one.
            CampaignNotFoundError: If the campaign is unknown.
            UnknownCampaignCellError: If the cell is not one of the campaign's grid.
        """
        self._run(self.command_of(arguments))

    @staticmethod
    def command_of(arguments: Mapping[str, JobArgument]) -> RunCampaignCellCommand:
        """The coordinates a job carries, read back into a command.

        Public because reading a message back is what this entry point is for, and what a
        queue delivers is the one input nobody in this process wrote. Every coordinate is
        required, the budget included: a message that lost it would otherwise be run at a
        budget nobody asked for rather than refused.

        Raises:
            ValueError: If a coordinate is missing or is not what it has to be.
            InvalidLabelBudgetError: If the budget is neither a count nor the word for every
                window there is.
        """
        campaign, candidate = arguments.get("campaign"), arguments.get("candidate")
        budget, seed = arguments.get("budget"), arguments.get("seed")
        if not isinstance(campaign, str) or not isinstance(candidate, str):
            raise ValueError("a cell names its campaign and its candidate")
        if not isinstance(budget, str):
            raise ValueError("a cell names the budget of labels it learns from")
        if not isinstance(seed, int):
            raise ValueError("a cell names the seed of its repeat")
        return RunCampaignCellCommand(
            campaign=CampaignId.parse(campaign),
            cell=CampaignCell(
                candidate=CandidateRef(candidate),
                budget=LabelBudget.parse(budget),
                seed=seed,
            ),
        )
