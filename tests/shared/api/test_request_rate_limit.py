"""The allowance one caller has on the open routes, counted in this process."""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException, Request

from emblema.shared.api.request_rate_limit import RequestRateLimit


class Ticking:
    """A monotonic clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def request_from(host: str | None) -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "headers": [],
        "client": None if host is None else (host, 4321),
    }
    return Request(scope)


def limit(per_minute: int = 3, remembered: int = 2) -> tuple[Callable[[Request], None], Ticking]:
    clock = Ticking()
    rate_limit = RequestRateLimit(
        requests_per_minute=per_minute, clients_remembered=remembered, clock=clock
    )
    return (lambda request: asyncio.run(rate_limit(request))), clock


def test_a_caller_may_burst_the_whole_allowance_and_is_then_told_when_to_come_back() -> None:
    admit, _ = limit(per_minute=3)
    for _ in range(3):
        admit(request_from("10.0.0.1"))

    with pytest.raises(HTTPException) as refused:
        admit(request_from("10.0.0.1"))

    assert refused.value.status_code == 429
    # Three a minute is one every twenty seconds; the bucket is empty, so a whole token is a
    # whole twenty seconds away.
    assert refused.value.headers == {"Retry-After": "20"}
    assert "3 requests a minute" in refused.value.detail


def test_the_allowance_comes_back_at_a_steady_rate() -> None:
    admit, clock = limit(per_minute=3)
    for _ in range(3):
        admit(request_from("10.0.0.1"))

    clock.now += 19.0
    with pytest.raises(HTTPException) as refused:
        admit(request_from("10.0.0.1"))
    assert refused.value.headers == {"Retry-After": "1"}

    clock.now += 1.0
    admit(request_from("10.0.0.1"))


def test_the_allowance_never_grows_beyond_a_minutes_worth() -> None:
    admit, clock = limit(per_minute=2)
    admit(request_from("10.0.0.1"))
    clock.now += 3600.0

    admit(request_from("10.0.0.1"))
    admit(request_from("10.0.0.1"))
    with pytest.raises(HTTPException):
        admit(request_from("10.0.0.1"))


def test_each_caller_has_an_allowance_of_its_own() -> None:
    admit, _ = limit(per_minute=1)
    admit(request_from("10.0.0.1"))

    admit(request_from("10.0.0.2"))
    with pytest.raises(HTTPException):
        admit(request_from("10.0.0.1"))


def test_the_least_recently_seen_caller_is_forgotten_which_only_forgives() -> None:
    admit, _ = limit(per_minute=1, remembered=2)
    admit(request_from("10.0.0.1"))
    admit(request_from("10.0.0.2"))
    admit(request_from("10.0.0.3"))

    # The second caller is still remembered as spent; the first was forgotten to make room for
    # the third, so it starts afresh.
    with pytest.raises(HTTPException):
        admit(request_from("10.0.0.2"))
    admit(request_from("10.0.0.1"))


def test_a_caller_seen_again_is_remembered_over_one_seen_once() -> None:
    admit, _ = limit(per_minute=2, remembered=2)
    admit(request_from("10.0.0.1"))
    admit(request_from("10.0.0.2"))
    admit(request_from("10.0.0.1"))
    admit(request_from("10.0.0.3"))

    # Seeing the first again made the second the stale one.
    with pytest.raises(HTTPException):
        admit(request_from("10.0.0.1"))
    admit(request_from("10.0.0.2"))


def test_requests_without_a_peer_count_as_one_caller() -> None:
    admit, _ = limit(per_minute=1)
    admit(request_from(None))

    with pytest.raises(HTTPException):
        admit(request_from(None))


@pytest.mark.parametrize(("per_minute", "remembered"), [(0, 1), (1, 0)])
def test_an_allowance_of_nothing_is_refused_when_the_limit_is_built(
    per_minute: int, remembered: int
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        RequestRateLimit(requests_per_minute=per_minute, clients_remembered=remembered)
