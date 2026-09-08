from emblema.shared.adapters.in_memory.event_subscriber import InMemoryEventSubscriber
from emblema.shared.events.domain_event import DomainEvent


class InMemoryEventPublisher:
    """Synchronous in-process publisher: every matching handler has run when ``publish`` returns.

    A handler's exception propagates to the publisher and stops the dispatch, so a use case that
    publishes inside a transaction fails together with its subscribers instead of committing half
    of the change.
    """

    def __init__(self, subscriptions: InMemoryEventSubscriber) -> None:
        self._subscriptions = subscriptions

    def publish(self, event: DomainEvent) -> None:
        for handler in self._subscriptions.handlers_for(event):
            handler(event)
