from typing import Protocol

from emblema.shared.events.domain_event import DomainEvent


class EventPublisher(Protocol):
    """Outbound side of the event mechanism: the only side a use case depends on.

    ``publish`` hands the event to every handler subscribed to its type or to a base of its
    type. Whether delivery completes before the call returns, and whether a failing handler
    fails the publisher, is a property of the adapter, not a promise of the port.
    """

    def publish(self, event: DomainEvent) -> None: ...
