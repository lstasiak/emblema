"""What any place orders of campaign cells pass through owes, whether a dictionary or a bucket.

What is placed comes back equal, an order is never read as a result nor a result as an order,
and the same result reported twice is one reference. Progress and the result to accept read
back alike but are stored apart, so the store can let the first expire and keep the second.
The store-backed adapter round-trips through JSON, so equality here is the codec's too: a
task, a candidate and a unit's error read back as exactly what was written.
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

from emblema.evaluation.adapters.handoff.artifact_store_campaign_handoff import (
    ArtifactStoreCampaignHandoff,
)
from emblema.evaluation.adapters.in_memory.campaign_handoff import InMemoryCampaignHandoff
from emblema.evaluation.domain.exceptions import UnreadableCampaignDocumentError
from emblema.evaluation.domain.handoff.campaign_order import CampaignOrder
from emblema.evaluation.domain.handoff.campaign_order_result import CampaignOrderResult
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout
from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.evaluation.support import CAMPAIGN, artifact, campaign, result, selection, task

ADAPTERS: dict[str, Callable[[], CampaignHandoff]] = {
    "in memory": InMemoryCampaignHandoff,
    "artifact store": lambda: ArtifactStoreCampaignHandoff(InMemoryArtifactStore()),
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def handoff(request: pytest.FixtureRequest) -> CampaignHandoff:
    made: CampaignHandoff = request.param()
    return made


def an_order() -> CampaignOrder:
    grid = campaign()
    return CampaignOrder(
        campaign=CAMPAIGN,
        task=task(),
        evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending()),
        git_commit="abc123",
    )


def a_result(order: ArtifactRef) -> CampaignOrderResult:
    first, second = an_order().cells[:2]
    return CampaignOrderResult(
        order=order,
        campaign=CAMPAIGN,
        git_commit="abc123",
        results=(
            result(first.candidate, first.budget, first.seed, (1.5, 0.1 + 0.2)),
            result(second.candidate, second.budget, second.seed, (2.0,), artifact=artifact("kept")),
        ),
    )


def test_an_order_comes_back_as_it_was_placed(handoff: CampaignHandoff) -> None:
    placed = an_order()

    assert handoff.read_order(handoff.place(placed)) == placed


def test_a_result_comes_back_as_it_was_reported_to_the_last_bit(handoff: CampaignHandoff) -> None:
    reported = a_result(handoff.place(an_order()))

    assert handoff.read_result(handoff.report(reported)) == reported


def test_progress_comes_back_as_it_was_reported(handoff: CampaignHandoff) -> None:
    reported = a_result(handoff.place(an_order()))

    assert handoff.read_progress(handoff.report_progress(reported)) == reported


def test_progress_no_longer_stored_reads_as_none(handoff: CampaignHandoff) -> None:
    gone = ArtifactRef("transient/sha256/gone", Checksum.of_bytes(b"gone"))

    assert handoff.read_progress(gone) is None


def test_progress_that_is_not_a_result_is_still_refused(handoff: CampaignHandoff) -> None:
    with pytest.raises(UnreadableCampaignDocumentError):
        handoff.read_progress(handoff.place(an_order()))


def test_the_same_result_reported_twice_is_one_reference(handoff: CampaignHandoff) -> None:
    reported = a_result(handoff.place(an_order()))

    assert handoff.report(reported) == handoff.report(reported)
    assert handoff.report_progress(reported) == handoff.report_progress(reported)


def test_progress_and_the_result_to_accept_are_kept_apart(handoff: CampaignHandoff) -> None:
    reported = a_result(handoff.place(an_order()))

    progress = handoff.report_progress(reported)
    final = handoff.report(reported)

    assert progress != final
    assert handoff.read_result(progress) == handoff.read_result(final)


def test_in_the_store_progress_is_transient_and_the_result_to_accept_durable() -> None:
    store = InMemoryArtifactStore()
    handoff = ArtifactStoreCampaignHandoff(store)
    reported = a_result(handoff.place(an_order()))

    progress = handoff.report_progress(reported)
    final = handoff.report(reported)

    assert progress.key.startswith(f"{Retention.TRANSIENT}/")
    assert final.key.startswith(f"{Retention.DURABLE}/")
    assert progress.checksum == final.checksum


def test_an_order_is_not_read_as_a_result_nor_a_result_as_an_order(
    handoff: CampaignHandoff,
) -> None:
    order = handoff.place(an_order())
    reported = handoff.report(a_result(order))

    with pytest.raises(UnreadableCampaignDocumentError):
        handoff.read_result(order)
    with pytest.raises(UnreadableCampaignDocumentError):
        handoff.read_order(reported)


def test_a_reference_to_nothing_is_reported_as_missing(handoff: CampaignHandoff) -> None:
    with pytest.raises(ArtifactNotFoundError):
        handoff.read_order(ArtifactRef("durable/absent", Checksum.of_bytes(b"absent")))


def test_an_order_of_selection_cells_carries_how_the_tuning_side_is_divided(
    handoff: CampaignHandoff,
) -> None:
    grid = replace(selection().campaign, recorded=(), completed_at=None)
    placed = CampaignOrder(
        campaign=grid.campaign_id,
        task=task(),
        evaluations=tuple(grid.evaluation_of(cell) for cell in grid.pending()),
        git_commit="abc123",
    )

    read = handoff.read_order(handoff.place(placed))

    assert read == placed
    assert {evaluation.holdout for evaluation in read.evaluations} == {InnerHoldout(one_in=5)}
