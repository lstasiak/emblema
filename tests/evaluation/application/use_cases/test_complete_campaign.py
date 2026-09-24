"""A campaign closed only once its verdict reads, and a selection closed announcing nothing."""

from dataclasses import replace

import pytest

from emblema.evaluation.adapters.in_memory.evaluation_campaign_repository import (
    InMemoryEvaluationCampaignRepository,
)
from emblema.evaluation.application.assemblers.campaign_completed_assembler import (
    CampaignCompletedAssembler,
)
from emblema.evaluation.application.use_cases.complete_campaign import (
    CompleteCampaign,
    CompleteCampaignCommand,
)
from emblema.evaluation.contracts.events import CampaignCompleted
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import (
    InvalidPairedUnitErrorsError,
    SelectionHasNoVerdictError,
)
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.evaluation.support import OPENED_AT, SELECTED_BY, campaign, result, selection


class Closing:
    def __init__(self, stored: EvaluationCampaign) -> None:
        self.published: list[CampaignCompleted] = []
        subscriptions = InMemoryEventSubscriber()
        subscriptions.subscribe(CampaignCompleted, self.published.append)
        self.campaigns = InMemoryEvaluationCampaignRepository()
        self.campaigns.save(stored, seen=0)
        self.complete = CompleteCampaign(
            self.campaigns,
            CampaignCompletedAssembler(),
            FixedClock(OPENED_AT),
            SequentialIdGenerator(),
            InMemoryEventPublisher(subscriptions),
        )


def whole(errors_of: dict[int, tuple[float, ...]] | None = None) -> EvaluationCampaign:
    """A comparison with every cell recorded and still open."""
    grid = campaign()
    for cell in grid.design.cells():
        errors = (errors_of or {}).get(cell.seed, (3.0, 4.0, 5.0))
        grid = grid.record(result(cell.candidate, cell.budget, cell.seed, errors))
    return grid


def test_a_selection_closes_and_announces_nothing() -> None:
    closing = Closing(replace(selection(), completed_at=None))

    announced = closing.complete(CompleteCampaignCommand(campaign=SELECTED_BY))

    assert announced is None
    assert closing.published == []
    assert closing.campaigns.get(SELECTED_BY).is_finished


def test_a_selection_is_asked_what_it_chose_never_what_it_concluded() -> None:
    with pytest.raises(SelectionHasNoVerdictError, match="which variant it chose"):
        selection().verdict()


def test_a_selection_keeps_no_fitted_candidate() -> None:
    grid = selection()

    assert not any(grid.design.retains(cell) for cell in grid.design.cells())


def test_a_verdict_that_cannot_be_read_leaves_the_campaign_open_to_be_closed_again() -> None:
    # The second repeat scores one unit fewer, so no pairing over the repeats exists.
    broken = whole({2: (3.0, 4.0)})
    closing = Closing(broken)

    with pytest.raises(InvalidPairedUnitErrorsError):
        closing.complete(CompleteCampaignCommand(campaign=broken.campaign_id))

    assert not closing.campaigns.get(broken.campaign_id).is_finished
    assert closing.published == []


def test_a_comparison_whose_verdict_reads_is_closed_and_announced() -> None:
    grid = whole()
    closing = Closing(grid)

    announced = closing.complete(CompleteCampaignCommand(campaign=grid.campaign_id))

    assert closing.published == [announced]
    assert closing.campaigns.get(grid.campaign_id).is_finished
