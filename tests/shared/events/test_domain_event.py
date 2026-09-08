from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from emblema.shared.events.domain_event import DomainEvent, EventId
from emblema.shared.kernel.timestamps import UtcDateTime

AT = UtcDateTime(datetime(2026, 9, 8, 12, tzinfo=UTC))
EVENT_ID = EventId(UUID(int=1))


@dataclass(frozen=True, kw_only=True)
class SomethingHappened(DomainEvent):
    subject: str


def test_event_fields_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        SomethingHappened(EVENT_ID, AT, "corpus")  # type: ignore[call-arg]


def test_event_is_immutable() -> None:
    event = SomethingHappened(event_id=EVENT_ID, occurred_at=AT, subject="corpus")

    with pytest.raises(FrozenInstanceError):
        event.subject = "other"  # type: ignore[misc]
