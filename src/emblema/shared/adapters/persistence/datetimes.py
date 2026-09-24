"""The one way a timestamp read from the database becomes the kernel's UTC value.

The database answers in the session's time zone, and the value object requires offset zero
rather than converting, so the boundary that produced the value is the one to normalise it. Every
record class of every context reads its timestamps through this so that the rule is stated once.
"""

from datetime import UTC, datetime

from emblema.shared.kernel.exceptions import InvalidUtcDateTimeError
from emblema.shared.kernel.timestamps import UtcDateTime


def as_utc(read: datetime) -> UtcDateTime:
    """The instant a stored timestamp came back as, expressed in UTC.

    Raises:
        InvalidUtcDateTimeError: If the value came back without a time zone, since a naive
            datetime names no instant and converting it would guess one.
    """
    if read.tzinfo is None:
        raise InvalidUtcDateTimeError("a stored timestamp came back without a time zone")
    return UtcDateTime(read.astimezone(UTC))
