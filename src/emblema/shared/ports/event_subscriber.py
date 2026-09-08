from collections.abc import Callable
from typing import Protocol

from emblema.shared.events.domain_event import DomainEvent


class EventSubscriber(Protocol):
    """Inbound side of the event mechanism: where handlers are registered.

    Used by the composition root and by the modules of a context that translate foreign events
    into its own operations, never by a use case. A handler registered for a type receives every
    event of that type or of a subtype, so a handler for ``DomainEvent`` sees everything.
    Handlers of one event run in subscription order.
    """

    def subscribe[E: DomainEvent](
        self, event_type: type[E], handler: Callable[[E], None]
    ) -> None: ...
