from typing import Any

import pytest

from emblema.evaluation.domain.exceptions import InvalidCampaignOrderResultError
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.evaluation.support import CAMPAIGN, CONTROL, result

REF = ArtifactRef("durable/order", Checksum.of_bytes(b"order"))
ONE = result(CONTROL, LabelBudget.of(50), 1, (1.0,))
TWO = result(CONTROL, LabelBudget.of(50), 2, (2.0,))


def reported(**overrides: Any) -> CampaignOrderResult:
    stated: dict[str, Any] = {
        "order": REF,
        "campaign": CAMPAIGN,
        "git_commit": "abc123",
        "results": (ONE,),
    }
    return CampaignOrderResult(**(stated | overrides))


def test_a_result_answering_nothing_is_refused() -> None:
    with pytest.raises(InvalidCampaignOrderResultError, match="at least one cell"):
        reported(results=())


def test_a_cell_answered_twice_is_refused_including_by_adding_it_again() -> None:
    with pytest.raises(InvalidCampaignOrderResultError, match="twice"):
        reported().with_result(ONE)


def test_a_blank_commit_is_refused() -> None:
    with pytest.raises(InvalidCampaignOrderResultError, match="git_commit"):
        reported(git_commit="")


def test_a_cell_answered_later_joins_those_before_it_in_the_order_run() -> None:
    grown = reported().with_result(TWO)

    assert grown.results == (ONE, TWO)
    assert grown.cells == frozenset({ONE.cell, TWO.cell})
