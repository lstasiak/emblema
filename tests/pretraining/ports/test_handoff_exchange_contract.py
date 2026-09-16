"""Contract of the HandoffExchange port, run against every adapter.

What is placed comes back unchanged under the reference given for it; the same document placed
twice has one reference; a reference names one kind of document and is refused as the other.
"""

from collections.abc import Callable

import pytest

from emblema.pretraining.adapters.handoff.artifact_store_handoff_exchange import (
    ArtifactStoreHandoffExchange,
)
from emblema.pretraining.adapters.in_memory.handoff_exchange import InMemoryHandoffExchange
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.pretraining.ports.handoff_exchange import HandoffExchange
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.support.experiments import WEIGHTS, epoch_outcome
from tests.support.handoff import CHECKPOINT, order, result

ADAPTERS: dict[str, Callable[[], HandoffExchange]] = {
    "in_memory": InMemoryHandoffExchange,
    "artifact_store": lambda: ArtifactStoreHandoffExchange(InMemoryArtifactStore()),
}
UNKNOWN = ArtifactRef("durable/nowhere", Checksum.of_bytes(b"nothing under this"))


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def exchange(request: pytest.FixtureRequest) -> HandoffExchange:
    build: Callable[[], HandoffExchange] = request.param
    return build()


def test_an_order_placed_reads_back_unchanged(exchange: HandoffExchange) -> None:
    placed = order(run="the-run")

    ref = exchange.place(placed)

    assert exchange.read_order(ref) == placed


def test_a_result_reported_reads_back_unchanged_with_every_reference(
    exchange: HandoffExchange,
) -> None:
    epochs = (
        epoch_outcome(2, checkpoint=CHECKPOINT),
        epoch_outcome(3, checkpoint=CHECKPOINT, backbone=WEIGHTS),
    )
    reported = result(
        resumed_from=CHECKPOINT, outcome=TrainingOutcome(backbone=WEIGHTS, epochs=epochs)
    )

    ref = exchange.report(reported)

    assert exchange.read_result(ref) == reported


def test_the_same_order_placed_twice_has_one_reference(exchange: HandoffExchange) -> None:
    assert exchange.place(order()) == exchange.place(order())
    assert exchange.place(order()) != exchange.place(order(run="another"))


def test_a_reference_names_one_kind_of_document(exchange: HandoffExchange) -> None:
    placed, reported = exchange.place(order()), exchange.report(result())

    with pytest.raises(UnreadableHandoffDocumentError):
        exchange.read_result(placed)
    with pytest.raises(UnreadableHandoffDocumentError):
        exchange.read_order(reported)


def test_a_document_that_is_not_there_is_reported(exchange: HandoffExchange) -> None:
    with pytest.raises(ArtifactNotFoundError):
        exchange.read_order(UNKNOWN)
    with pytest.raises(ArtifactNotFoundError):
        exchange.read_result(UNKNOWN)
