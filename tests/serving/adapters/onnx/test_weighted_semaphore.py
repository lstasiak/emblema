import threading
import time

import pytest

from emblema.serving.adapters.onnx.weighted_semaphore import WeightedSemaphore
from emblema.serving.domain.exceptions import InferenceBusyError


def test_weight_is_held_for_the_block_and_given_back_after() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=1)

    with semaphore.reserve(4):
        assert semaphore.in_use == 4
        with semaphore.reserve(6):
            assert semaphore.in_use == 10

    assert semaphore.in_use == 0


def test_weight_is_given_back_when_the_guarded_work_fails() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=1)

    with pytest.raises(RuntimeError), semaphore.reserve(10):
        raise RuntimeError("the graph failed")

    assert semaphore.in_use == 0
    with semaphore.reserve(10):
        pass


def test_a_caller_that_cannot_be_admitted_in_time_is_refused_and_leaves_no_trace() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=0.05)

    with (
        semaphore.reserve(8),
        pytest.raises(InferenceBusyError, match=r"none could be started within 0\.05 s"),
        semaphore.reserve(5),
    ):
        pass

    assert semaphore.in_use == 0
    assert semaphore.waiting == 0


def test_a_caller_waits_for_weight_to_be_given_back_rather_than_being_refused() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=5)
    admitted = threading.Event()

    def wanting() -> None:
        with semaphore.reserve(6):
            admitted.set()

    with semaphore.reserve(8):
        waiter = threading.Thread(target=wanting)
        waiter.start()
        assert not admitted.wait(0.1)
    waiter.join(timeout=5)

    assert admitted.is_set()


def test_a_heavy_caller_at_the_head_is_not_overtaken_by_light_ones_that_would_fit() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=5)
    order: list[str] = []
    heavy_waiting = threading.Event()

    def run(name: str, weight: int) -> None:
        with semaphore.reserve(weight):
            order.append(name)

    with semaphore.reserve(6):
        heavy = threading.Thread(target=run, args=("heavy", 8))
        heavy.start()
        while semaphore.waiting < 1:
            time.sleep(0.005)
        heavy_waiting.set()
        light = threading.Thread(target=run, args=("light", 2))
        light.start()
        while semaphore.waiting < 2:
            time.sleep(0.005)
        assert order == []
    heavy.join(timeout=5)
    light.join(timeout=5)

    assert order == ["heavy", "light"]


def test_what_runs_at_once_never_weighs_more_than_the_semaphore_holds() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=10)
    peak = 0
    lock = threading.Lock()

    def work(weight: int) -> None:
        nonlocal peak
        with semaphore.reserve(weight):
            with lock:
                peak = max(peak, semaphore.in_use)
            time.sleep(0.01)

    threads = [threading.Thread(target=work, args=(1 + n % 7,)) for n in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert 0 < peak <= 10
    assert semaphore.in_use == 0


@pytest.mark.parametrize("weight", [0, -1, 11])
def test_a_weight_outside_what_the_semaphore_holds_is_refused(weight: int) -> None:
    with (
        pytest.raises(ValueError, match="between 1 and 10"),
        WeightedSemaphore(10, wait_seconds=1).reserve(weight),
    ):
        pass


@pytest.mark.parametrize(
    ("capacity", "wait", "reason"),
    [(0, 1.0, "at least one unit"), (10, 0.0, "positive time")],
)
def test_a_semaphore_holds_something_and_a_caller_waits_some_time(
    capacity: int, wait: float, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        WeightedSemaphore(capacity, wait_seconds=wait)


def test_a_deadline_already_passed_refuses_at_once_even_where_the_weight_is_free() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=5)
    started = time.monotonic()

    with (
        semaphore.reserve(8),
        pytest.raises(InferenceBusyError),
        semaphore.reserve(5, until=time.monotonic() - 1),
    ):
        pass

    assert time.monotonic() - started < 1
    with semaphore.reserve(10, until=time.monotonic() + 1):
        pass


def test_several_reservations_under_one_deadline_wait_that_long_together() -> None:
    semaphore = WeightedSemaphore(10, wait_seconds=0.3)
    until = semaphore.deadline()
    started = time.monotonic()

    with semaphore.reserve(8):
        with semaphore.reserve(2, until=until):
            pass
        with pytest.raises(InferenceBusyError), semaphore.reserve(5, until=until):
            pass
        with pytest.raises(InferenceBusyError), semaphore.reserve(5, until=until):
            pass

    assert 0.3 <= time.monotonic() - started < 0.6
