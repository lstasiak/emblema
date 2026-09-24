from datetime import UTC, datetime, timedelta, timezone

import pytest

from emblema.shared.adapters.persistence.datetimes import as_utc
from emblema.shared.kernel.exceptions import InvalidUtcDateTimeError
from emblema.shared.kernel.timestamps import UtcDateTime

INSTANT = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_a_timestamp_answered_in_another_time_zone_is_the_same_instant_in_utc() -> None:
    answered = INSTANT.astimezone(timezone(timedelta(hours=2)))

    assert as_utc(answered) == UtcDateTime(INSTANT)


def test_a_timestamp_answered_in_utc_is_taken_as_it_is() -> None:
    assert as_utc(INSTANT) == UtcDateTime(INSTANT)


def test_a_timestamp_answered_without_a_time_zone_is_refused_rather_than_guessed() -> None:
    with pytest.raises(InvalidUtcDateTimeError, match="without a time zone"):
        as_utc(INSTANT.replace(tzinfo=None))
