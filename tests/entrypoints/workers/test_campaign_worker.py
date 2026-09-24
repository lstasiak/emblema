"""What a queue delivers, read back into the coordinates of a cell.

The one input nobody in the process wrote, so every coordinate is demanded rather than defaulted
and a message that lost one is refused instead of being run at a budget nobody asked for. The
same for whichever pool the process serves, which is why it is held to account apart from either.
"""

from collections.abc import Mapping

import pytest

from emblema.entrypoints.workers.campaign_worker import CampaignWorker
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.exceptions import InvalidLabelBudgetError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.jobs.job_argument import JobArgument

CAMPAIGN_TEXT = "00000000-0000-0000-0000-000000000002"


def test_a_job_is_read_back_into_the_coordinates_of_a_cell() -> None:
    command = CampaignWorker.command_of(
        {
            "campaign": CAMPAIGN_TEXT,
            "candidate": "full_fine_tuning",
            "budget": "200",
            "seed": 3,
        }
    )

    assert command.cell.candidate == CandidateRef("full_fine_tuning")
    assert command.cell.budget == LabelBudget.of(200)
    assert command.cell.seed == 3


def test_a_job_at_the_largest_budget_names_every_label_the_tuning_side_holds() -> None:
    command = CampaignWorker.command_of(
        {"campaign": CAMPAIGN_TEXT, "candidate": "lora", "budget": "all", "seed": 1}
    )

    assert command.cell.budget == LabelBudget.everything()


def test_a_baseline_reaches_the_worker_under_the_name_it_competes_by() -> None:
    command = CampaignWorker.command_of(
        {
            "campaign": CAMPAIGN_TEXT,
            "candidate": "boosted_trees_across_channels",
            "budget": "50",
            "seed": 1,
        }
    )

    assert command.cell.candidate == CandidateRef("boosted_trees_across_channels")


@pytest.mark.parametrize(
    "arguments",
    [
        {"candidate": "lora", "budget": "50", "seed": 1},
        {"campaign": CAMPAIGN_TEXT, "budget": "50", "seed": 1},
        {"campaign": CAMPAIGN_TEXT, "candidate": "lora", "seed": 1},
        {"campaign": CAMPAIGN_TEXT, "candidate": "lora", "budget": 50, "seed": 1},
        {"campaign": CAMPAIGN_TEXT, "candidate": "lora", "budget": "50", "seed": "first"},
    ],
)
def test_a_job_that_does_not_name_a_cell_is_refused(
    arguments: Mapping[str, JobArgument],
) -> None:
    with pytest.raises(ValueError, match="a cell names"):
        CampaignWorker.command_of(arguments)


def test_a_job_naming_a_budget_that_is_not_one_is_refused() -> None:
    with pytest.raises(InvalidLabelBudgetError):
        CampaignWorker.command_of(
            {
                "campaign": CAMPAIGN_TEXT,
                "candidate": "lora",
                "budget": "most of them",
                "seed": 1,
            }
        )
