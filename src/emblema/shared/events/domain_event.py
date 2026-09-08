"""The event base type and the identity every event carries."""

from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId
from emblema.shared.kernel.timestamps import UtcDateTime


@dataclass(frozen=True)
class EventId(EntityId):
    pass


@dataclass(frozen=True, kw_only=True)
class DomainEvent:
    """A fact that became true in one bounded context and is published for the others.

    Concrete events are frozen, keyword-only dataclasses in the ``contracts`` package of the
    context that publishes them, so a subscriber depends on the published language alone. The
    publishing use case stamps identity and instant from the injected ``IdGenerator`` and
    ``Clock``; the domain model never builds events itself.

    Attributes:
        event_id: Identity of this occurrence, so a subscriber can recognise a redelivery.
        occurred_at: When the fact became true in the publishing context.
    """

    event_id: EventId
    occurred_at: UtcDateTime
