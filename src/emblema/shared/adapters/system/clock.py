from datetime import UTC, datetime

from emblema.shared.kernel.timestamps import UtcDateTime


class SystemClock:
    def now(self) -> UtcDateTime:
        return UtcDateTime(datetime.now(UTC))
