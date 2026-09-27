"""Contract of the VerdictMemo port, against its one adapter.

A memo keeps what it was given under the campaign and the revision, answers nothing for what it
was not given, and may forget; the in-memory adapter forgets the least recently asked first.
"""

from uuid import UUID

import pytest

from emblema.evaluation.adapters.in_memory.verdict_memo import InMemoryVerdictMemo
from emblema.evaluation.contracts.identifiers import CampaignId
from tests.evaluation.support import CAMPAIGN, closed_campaign

OTHER = CampaignId(UUID(int=77))
VERDICT = closed_campaign().verdict()


def test_a_kept_verdict_is_recalled_under_its_campaign_and_revision() -> None:
    memo = InMemoryVerdictMemo(capacity=2)

    memo.keep(CAMPAIGN, 9, VERDICT)

    assert memo.recall(CAMPAIGN, 9) == VERDICT
    assert memo.recall(CAMPAIGN, 8) is None
    assert memo.recall(OTHER, 9) is None


def test_past_its_capacity_the_memo_forgets_the_least_recently_asked() -> None:
    memo = InMemoryVerdictMemo(capacity=2)
    memo.keep(CAMPAIGN, 1, VERDICT)
    memo.keep(OTHER, 1, VERDICT)
    memo.recall(CAMPAIGN, 1)

    memo.keep(CAMPAIGN, 2, VERDICT)

    assert memo.recall(OTHER, 1) is None
    assert memo.recall(CAMPAIGN, 1) == VERDICT
    assert memo.recall(CAMPAIGN, 2) == VERDICT


def test_a_memo_keeps_at_least_one_verdict() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InMemoryVerdictMemo(capacity=0)
