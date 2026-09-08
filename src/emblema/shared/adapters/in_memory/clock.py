from emblema.shared.kernel.timestamps import UtcDateTime


class FixedClock:
    """Clock that always reports the instant it was built with."""

    def __init__(self, at: UtcDateTime) -> None:
        self._at = at

    def now(self) -> UtcDateTime:
        return self._at
