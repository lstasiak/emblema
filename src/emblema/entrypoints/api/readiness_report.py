from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ReadinessReport:
    """What the readiness probe found: every dependency named, with whether it answered.

    Attributes:
        ready: Whether every dependency answered.
        checks: Each dependency's answer, ``ok`` or ``failed``; why is in the process's log,
            not in a body anyone may read.
    """

    ready: bool
    checks: Mapping[str, str]
