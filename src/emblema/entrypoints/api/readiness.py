import logging
from collections.abc import Callable, Mapping

from emblema.entrypoints.api.readiness_report import ReadinessReport

logger = logging.getLogger(__name__)

OK, FAILED = "ok", "failed"


class Readiness:
    """Asks each dependency the process cannot answer without whether it is reachable.

    Each check is a call that returns on success and raises on failure, composed by the root
    over the real adapters, so this holds no connection of its own and a test composes it over
    checks that answer as it says. A failure is logged with its cause and reported only by
    name: the probe is public, and what an unreachable database says about itself — its host,
    its user — is not.
    """

    def __init__(self, checks: Mapping[str, Callable[[], None]]) -> None:
        self._checks = dict(checks)

    def report(self) -> ReadinessReport:
        answers: dict[str, str] = {}
        for name, check in self._checks.items():
            try:
                check()
            except Exception:
                logger.warning("readiness check %r failed", name, exc_info=True)
                answers[name] = FAILED
            else:
                answers[name] = OK
        return ReadinessReport(
            ready=all(answer == OK for answer in answers.values()), checks=answers
        )
