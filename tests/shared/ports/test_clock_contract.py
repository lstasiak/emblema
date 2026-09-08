"""Contract of the Clock port, run against every adapter."""

from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.kernel.timestamps import UtcDateTime
from emblema.shared.ports.clock import Clock

AT = UtcDateTime(datetime(2026, 9, 8, 12, tzinfo=UTC))

ADAPTERS: dict[str, Callable[[], Clock]] = {
    "system": SystemClock,
    "fixed": lambda: FixedClock(AT),
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def clock(request: pytest.FixtureRequest) -> Clock:
    factory: Callable[[], Clock] = request.param
    return factory()


def test_now_is_a_utc_datetime(clock: Clock) -> None:
    assert isinstance(clock.now(), UtcDateTime)


def test_fixed_clock_reports_the_configured_instant() -> None:
    assert FixedClock(AT).now() == AT


def test_system_clock_reads_the_wall_clock() -> None:
    before = datetime.now(UTC)
    observed = SystemClock().now().value
    after = datetime.now(UTC)

    assert before <= observed <= after
