from datetime import UTC, datetime, timedelta, timezone

import pytest

from emblema.shared.kernel.exceptions import InvalidUtcDateTimeError
from emblema.shared.kernel.timestamps import UtcDateTime

NOON = datetime(2026, 9, 8, 12, tzinfo=UTC)


def test_accepts_aware_utc_datetime() -> None:
    assert UtcDateTime(NOON).value == NOON


def test_accepts_any_zone_with_zero_offset() -> None:
    zero_offset = timezone(timedelta(0), name="Z")

    assert UtcDateTime(NOON.replace(tzinfo=zero_offset)).value == NOON


def test_rejects_naive_datetime() -> None:
    with pytest.raises(InvalidUtcDateTimeError, match="aware"):
        UtcDateTime(NOON.replace(tzinfo=None))


def test_rejects_non_utc_offset() -> None:
    warsaw = timezone(timedelta(hours=2))

    with pytest.raises(InvalidUtcDateTimeError, match="UTC"):
        UtcDateTime(NOON.astimezone(warsaw))


def test_orders_chronologically() -> None:
    earlier = UtcDateTime(NOON)
    later = UtcDateTime(NOON + timedelta(seconds=1))

    assert earlier < later
    assert max(later, earlier) == later
