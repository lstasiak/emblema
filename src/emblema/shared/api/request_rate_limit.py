import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    """What one caller may still ask, and when that was last worked out."""

    tokens: float
    refilled_at: float


class RequestRateLimit:
    """Admits each caller a bounded number of requests a minute, refused with ``429`` beyond.

    A token bucket per caller: a minute's allowance is the bucket, every request takes one
    token, and tokens come back at a steady rate, so a caller may burst up to the allowance and
    sustain the rate after it. A refusal says when the next request will be admitted. The
    caller is its network address; the buckets live in this process, bounded in number, the
    least recently seen forgotten first — forgetting one only ever forgives.
    """

    def __init__(
        self,
        *,
        requests_per_minute: int,
        clients_remembered: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set the allowance.

        Args:
            requests_per_minute: The allowance, and the size of a burst.
            clients_remembered: How many callers' buckets the process keeps at once.
            clock: Seconds from an arbitrary origin, monotonic; the system's unless given.

        Raises:
            ValueError: If either count is not positive.
        """
        if requests_per_minute < 1:
            raise ValueError("a caller is allowed at least one request a minute")
        if clients_remembered < 1:
            raise ValueError("at least one caller is remembered")
        self._allowance = requests_per_minute
        self._rate = requests_per_minute / 60.0
        self._remembered = clients_remembered
        self._clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> None:
        """Admit the request or refuse it.

        Raises:
            HTTPException: With ``429`` and a ``Retry-After`` if the caller's allowance is spent.
        """
        wait = self._take(self._caller_of(request))
        if wait > 0:
            raise HTTPException(
                HTTPStatus.TOO_MANY_REQUESTS.value,
                detail=(
                    f"at most {self._allowance} requests a minute are admitted per caller; "
                    f"try again in {wait} seconds"
                ),
                headers={"Retry-After": str(wait)},
            )

    def _take(self, caller: str) -> int:
        """Spend one token of the caller's bucket; zero, or the whole seconds until one is back."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(caller)
            if bucket is None:
                bucket = _Bucket(tokens=float(self._allowance), refilled_at=now)
                self._buckets[caller] = bucket
                while len(self._buckets) > self._remembered:
                    self._buckets.popitem(last=False)
            else:
                self._buckets.move_to_end(caller)
                elapsed = max(0.0, now - bucket.refilled_at)
                bucket.tokens = min(float(self._allowance), bucket.tokens + elapsed * self._rate)
                bucket.refilled_at = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return 0
            return max(1, math.ceil((1.0 - bucket.tokens) / self._rate))

    @staticmethod
    def _caller_of(request: Request) -> str:
        # A request without a peer — one made in-process by a test client — is one caller.
        return request.client.host if request.client is not None else "unknown"
