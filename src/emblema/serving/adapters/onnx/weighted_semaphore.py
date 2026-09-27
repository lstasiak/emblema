import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

from emblema.serving.domain.exceptions import InferenceBusyError


class WeightedSemaphore:
    """Admits work by what it costs, in the order it arrived, and refuses what waits too long.

    A counting semaphore admits a number of callers; this admits a total weight, so several
    light batches run beside each other and one heavy batch runs beside few. Callers are served
    first come, first served: a heavy caller at the head is not overtaken by light ones that
    would fit, or it could wait for ever behind a stream of them. A caller that cannot be
    admitted within ``wait_seconds`` is refused rather than kept, because a service that queues
    without bound answers everyone late and is killed for its memory in the end. A caller that
    reserves several times for one piece of work passes the same deadline to each, so the work
    as a whole waits that long and not that long per reservation.

    The weight is released however the guarded work ends.
    """

    def __init__(self, capacity: int, *, wait_seconds: float) -> None:
        if capacity < 1:
            raise ValueError(f"a semaphore admits at least one unit of weight, got {capacity}")
        if wait_seconds <= 0:
            raise ValueError(f"a caller waits for a positive time, got {wait_seconds}")
        self._capacity = capacity
        self._wait_seconds = wait_seconds
        self._condition = threading.Condition()
        self._in_use = 0
        self._queue: deque[object] = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        """Weight held right now."""
        with self._condition:
            return self._in_use

    @property
    def waiting(self) -> int:
        """Callers waiting to be admitted right now."""
        with self._condition:
            return len(self._queue)

    def deadline(self) -> float:
        """When a caller starting to wait now would be refused, on the monotonic clock."""
        return time.monotonic() + self._wait_seconds

    @contextmanager
    def reserve(self, weight: int, *, until: float | None = None) -> Iterator[None]:
        """Hold ``weight`` for the duration of the block.

        Args:
            weight: What to hold.
            until: When to give up waiting, as ``deadline`` gives it; a caller starting now
                unless given.

        Raises:
            ValueError: If ``weight`` is not positive or is more than the semaphore holds.
            InferenceBusyError: If the weight could not be had by the deadline.
        """
        if not 1 <= weight <= self._capacity:
            raise ValueError(f"a weight is between 1 and {self._capacity}, got {weight}")
        self._acquire(weight, self.deadline() if until is None else until)
        try:
            yield
        finally:
            with self._condition:
                self._in_use -= weight
                self._condition.notify_all()

    def _acquire(self, weight: int, deadline: float) -> None:
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            try:
                while not (self._queue[0] is ticket and self._in_use + weight <= self._capacity):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise InferenceBusyError(
                            "the networks are running as much as they may and none could be "
                            f"started within {self._wait_seconds:g} s; "
                            f"{len(self._queue) - 1} more waiting"
                        )
                    self._condition.wait(remaining)
                self._in_use += weight
            finally:
                self._queue.remove(ticket)
                self._condition.notify_all()
