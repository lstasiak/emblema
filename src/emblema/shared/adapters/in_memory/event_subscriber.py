from collections.abc import Callable
from typing import Any

from emblema.shared.events.domain_event import DomainEvent


class InMemoryEventSubscriber:
    """Registry of handlers for in-process delivery: decides which handlers an event goes to.

    A handler registered for a type matches events of that type and of its subtypes; matches
    come back in subscription order. Invoking them is the publisher's job
    (``InMemoryEventPublisher``), which is why the two adapters share this object.
    """

    def __init__(self) -> None:
        # Handlers of different event types share one list, so its element type cannot be
        # narrower than Any; each pair was checked against its event type in `subscribe`.
        self._subscriptions: list[tuple[type[DomainEvent], Callable[[Any], None]]] = []

    def subscribe[E: DomainEvent](self, event_type: type[E], handler: Callable[[E], None]) -> None:
        self._subscriptions.append((event_type, handler))

    def handlers_for(self, event: DomainEvent) -> tuple[Callable[[DomainEvent], None], ...]:
        return tuple(
            handler for event_type, handler in self._subscriptions if isinstance(event, event_type)
        )
