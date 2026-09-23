"""What the store-backed exchange adds to the contract: durable documents, foreign bytes refused."""

import pytest

from emblema.pretraining.adapters.handoff.artifact_store_handoff_exchange import (
    ArtifactStoreHandoffExchange,
)
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.retention import Retention
from tests.support.handoff import order, result


def test_orders_and_results_are_durable_artifacts() -> None:
    store = InMemoryArtifactStore()
    exchange = ArtifactStoreHandoffExchange(store)

    placed, reported = exchange.place(order()), exchange.report(result())

    assert placed.key.startswith(f"{Retention.DURABLE}/")
    assert reported.key.startswith(f"{Retention.DURABLE}/")
    assert store.exists(placed)
    assert store.exists(reported)


def test_bytes_that_are_neither_document_are_refused() -> None:
    store = InMemoryArtifactStore()
    ref = store.put(b"neither an order nor a result", Retention.DURABLE)
    exchange = ArtifactStoreHandoffExchange(store)

    with pytest.raises(UnreadableHandoffDocumentError):
        exchange.read_order(ref)
    with pytest.raises(UnreadableHandoffDocumentError):
        exchange.read_result(ref)
