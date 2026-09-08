from typing import Protocol

from emblema.shared.kernel.timestamps import UtcDateTime


class Clock(Protocol):
    """Source of the current time, injected so that domain time never comes from the wall clock."""

    def now(self) -> UtcDateTime: ...
