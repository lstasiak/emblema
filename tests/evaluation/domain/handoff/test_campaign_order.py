from dataclasses import replace
from typing import Any
from uuid import UUID

import pytest

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId
from emblema.evaluation.domain.exceptions import (
    CampaignOrderRejectedError,
    InvalidCampaignOrderError,
)
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.evaluation.support import CAMPAIGN, CONTENDER, CONTROL, campaign, result, task

REF = ArtifactRef("durable/order", Checksum.of_bytes(b"order"))


def order(**overrides: Any) -> CampaignOrder:
    grid = campaign()
    stated = CampaignOrder(
        campaign=CAMPAIGN,
        task=task(),
        evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending()),
        git_commit="abc123",
    )
    return replace(stated, **overrides)


def test_an_order_of_no_cell_is_refused() -> None:
    with pytest.raises(InvalidCampaignOrderError, match="names no cell"):
        order(evaluations=())


def test_an_order_naming_a_cell_twice_is_refused() -> None:
    once = order().evaluations[0]

    with pytest.raises(InvalidCampaignOrderError, match="twice"):
        order(evaluations=(once, once))


def test_a_cell_of_another_task_is_refused() -> None:
    stray = replace(order().evaluations[0], task=TaskId(UUID(int=77)))

    with pytest.raises(InvalidCampaignOrderError, match="over"):
        order(evaluations=(stray,))


def test_cells_needing_the_training_stack_never_share_an_order_with_those_that_must_not() -> None:
    neural, other = order().evaluations[:2]
    classical = replace(
        other,
        declared=replace(
            other.declared, kind=CandidateKind.CLASSICAL, budget=None, starts_from=None
        ),
    )

    with pytest.raises(InvalidCampaignOrderError, match="training stack"):
        order(evaluations=(neural, classical))


@pytest.mark.parametrize("commit", ["", " abc123"])
def test_a_commit_that_is_blank_or_padded_is_refused(commit: str) -> None:
    with pytest.raises(InvalidCampaignOrderError, match="git_commit"):
        order(git_commit=commit)


def test_a_run_on_other_code_is_refused() -> None:
    with pytest.raises(CampaignOrderRejectedError, match="def456"):
        order().require_commit("def456")


def test_a_result_answering_only_cells_of_the_order_is_held_to_it() -> None:
    first = order().cells[0]
    answered = result(first.candidate, first.budget, first.seed, (1.0,))

    order().require_answered_by(
        CampaignOrderResult(order=REF, campaign=CAMPAIGN, git_commit="abc123", results=(answered,))
    )


@pytest.mark.parametrize(
    ("campaign_id", "commit"),
    [(CampaignId(UUID(int=5)), "abc123"), (CAMPAIGN, "def456")],
    ids=["another campaign", "other code"],
)
def test_a_result_of_another_campaign_or_other_code_is_refused(
    campaign_id: CampaignId, commit: str
) -> None:
    first = order().cells[0]
    answered = result(first.candidate, first.budget, first.seed, (1.0,))

    with pytest.raises(CampaignOrderRejectedError, match="does not answer"):
        order().require_answered_by(
            CampaignOrderResult(
                order=REF, campaign=campaign_id, git_commit=commit, results=(answered,)
            )
        )


def test_the_pool_follows_from_what_the_cells_are() -> None:
    assert order().needs_the_ml_stack is True
    assert {cell.candidate for cell in order().cells} == {CONTROL, CONTENDER}
    assert all(cell.budget in (LabelBudget.of(50), LabelBudget.of(200)) for cell in order().cells)
