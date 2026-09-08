"""Contract of the event ports, run against every publisher/subscriber pair of adapters."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import UUID

import pytest

from emblema.shared.adapters.in_memory.event_publisher import InMemoryEventPublisher
from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.events.domain_event import DomainEvent, EventId
from emblema.shared.kernel.timestamps import UtcDateTime
from emblema.shared.ports.event_publisher import EventPublisher
from emblema.shared.ports.event_subscriber import EventSubscriber

AT = UtcDateTime(datetime(2026, 9, 8, 12, tzinfo=UTC))
EVENT_ID = EventId(UUID(int=1))


class EventPorts(NamedTuple):
    """Both ends of one delivery mechanism, as the composition root wires them."""

    publisher: EventPublisher
    subscriber: EventSubscriber


@dataclass(frozen=True, kw_only=True)
class OrderPlaced(DomainEvent):
    order: str


@dataclass(frozen=True, kw_only=True)
class PriorityOrderPlaced(OrderPlaced):
    pass


@dataclass(frozen=True, kw_only=True)
class OrderCancelled(DomainEvent):
    order: str


def placed(order: str = "a") -> OrderPlaced:
    return OrderPlaced(event_id=EVENT_ID, occurred_at=AT, order=order)


def cancelled(order: str = "a") -> OrderCancelled:
    return OrderCancelled(event_id=EVENT_ID, occurred_at=AT, order=order)


def in_memory() -> EventPorts:
    subscriptions = InMemoryEventSubscriber()
    return EventPorts(InMemoryEventPublisher(subscriptions), subscriptions)


ADAPTERS: dict[str, Callable[[], EventPorts]] = {"in_memory": in_memory}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def ports(request: pytest.FixtureRequest) -> EventPorts:
    factory: Callable[[], EventPorts] = request.param
    return factory()


def test_handler_receives_events_of_its_type(ports: EventPorts) -> None:
    received: list[OrderPlaced] = []
    ports.subscriber.subscribe(OrderPlaced, received.append)

    ports.publisher.publish(placed("a"))
    ports.publisher.publish(placed("b"))

    assert received == [placed("a"), placed("b")]


def test_handler_ignores_other_event_types(ports: EventPorts) -> None:
    received: list[OrderPlaced] = []
    ports.subscriber.subscribe(OrderPlaced, received.append)

    ports.publisher.publish(cancelled())

    assert received == []


def test_handler_of_a_base_type_receives_every_subtype(ports: EventPorts) -> None:
    received: list[DomainEvent] = []
    ports.subscriber.subscribe(DomainEvent, received.append)

    ports.publisher.publish(placed())
    ports.publisher.publish(cancelled())

    assert received == [placed(), cancelled()]


def test_handler_of_a_subtype_does_not_receive_the_base_type(ports: EventPorts) -> None:
    received: list[PriorityOrderPlaced] = []
    ports.subscriber.subscribe(PriorityOrderPlaced, received.append)

    ports.publisher.publish(placed())

    assert received == []


def test_handlers_run_in_subscription_order(ports: EventPorts) -> None:
    calls: list[str] = []
    ports.subscriber.subscribe(OrderPlaced, lambda _: calls.append("first"))
    ports.subscriber.subscribe(DomainEvent, lambda _: calls.append("second"))
    ports.subscriber.subscribe(OrderPlaced, lambda _: calls.append("third"))

    ports.publisher.publish(placed())

    assert calls == ["first", "second", "third"]


def test_publishing_without_handlers_is_allowed(ports: EventPorts) -> None:
    ports.publisher.publish(placed())


def test_in_memory_publisher_stops_dispatch_at_the_failing_handler() -> None:
    subscriptions = InMemoryEventSubscriber()
    publisher = InMemoryEventPublisher(subscriptions)
    reached: list[OrderPlaced] = []

    def fail(_: OrderPlaced) -> None:
        raise RuntimeError("projection unavailable")

    subscriptions.subscribe(OrderPlaced, fail)
    subscriptions.subscribe(OrderPlaced, reached.append)

    with pytest.raises(RuntimeError, match="projection unavailable"):
        publisher.publish(placed())

    assert reached == []
