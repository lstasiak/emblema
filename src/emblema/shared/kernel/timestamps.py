from dataclasses import dataclass
from datetime import datetime, timedelta

from emblema.shared.kernel.exceptions import InvalidUtcDateTimeError


@dataclass(frozen=True, order=True)
class UtcDateTime:
    """Point in time that is guaranteed to be timezone-aware and expressed in UTC.

    The constructor rejects naive datetimes and any non-zero offset instead of converting them:
    normalisation belongs to the boundary that produced the value, so a wrong timezone fails
    where it originates rather than being silently reinterpreted.

    Attributes:
        value: The wrapped datetime, aware, offset zero.
    """

    value: datetime

    def __post_init__(self) -> None:
        offset = self.value.utcoffset()
        if offset is None:
            raise InvalidUtcDateTimeError("datetime must be timezone-aware")
        if offset != timedelta(0):
            raise InvalidUtcDateTimeError(f"datetime must be in UTC, got offset {offset}")
