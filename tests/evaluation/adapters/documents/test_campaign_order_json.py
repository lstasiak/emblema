import json
from dataclasses import replace

import pytest

from emblema.evaluation.adapters.documents.campaign_order_json import CampaignOrderJson
from emblema.evaluation.adapters.documents.campaign_order_result_json import (
    CampaignOrderResultJson,
)
from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from tests.evaluation.support import CAMPAIGN, campaign, selection, task

CODEC = CampaignOrderJson()


def an_order() -> CampaignOrder:
    grid = campaign()
    return CampaignOrder(
        campaign=CAMPAIGN,
        task=task(),
        evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending()),
        git_commit="abc123",
    )


def test_one_order_is_always_the_same_bytes() -> None:
    assert CODEC.encode(an_order()) == CODEC.encode(an_order())


@pytest.mark.parametrize(
    "content",
    [b"\xff", b"[]", b'{"format": "emblema.campaign-order", "version": 99}'],
    ids=["not text", "not an object", "another version"],
)
def test_bytes_that_are_not_an_order_of_this_version_are_refused(content: bytes) -> None:
    with pytest.raises(UnreadableCampaignDocumentError):
        CODEC.decode(content)


def test_an_order_missing_a_field_is_refused_naming_it() -> None:
    document = json.loads(CODEC.encode(an_order()))
    del document["task"]["manifest"]

    with pytest.raises(UnreadableCampaignDocumentError, match="manifest"):
        CODEC.decode(json.dumps(document).encode())


def test_an_order_that_states_something_no_order_can_be_is_refused() -> None:
    document = json.loads(CODEC.encode(an_order()))
    document["git_commit"] = " padded"

    with pytest.raises(UnreadableCampaignDocumentError, match="git_commit"):
        CODEC.decode(json.dumps(document).encode())


def test_an_order_of_a_fixed_division_carries_its_seed_and_one_placed_before_divides_afresh() -> (
    None
):
    grid = replace(selection().campaign, recorded=(), completed_at=None)
    fixed = replace(
        grid, design=replace(grid.design, inner_holdout=InnerHoldout(one_in=5, division_seed=101))
    )
    placed = replace(
        an_order(), evaluations=tuple(fixed.evaluation_of(cell) for cell in fixed.pending())
    )
    earlier = json.loads(
        CODEC.encode(
            replace(
                an_order(), evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending())
            )
        )
    )
    for evaluation in earlier["evaluations"]:
        del evaluation["division_seed"]

    assert CODEC.decode(CODEC.encode(placed)) == placed
    assert {e.holdout for e in CODEC.decode(json.dumps(earlier).encode()).evaluations} == {
        InnerHoldout(one_in=5)
    }


def test_a_result_codec_refuses_an_order() -> None:
    with pytest.raises(UnreadableCampaignDocumentError, match="campaign-order"):
        CampaignOrderResultJson().decode(CODEC.encode(an_order()))
