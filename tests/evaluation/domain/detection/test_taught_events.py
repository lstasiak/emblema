import pytest

from emblema.evaluation.domain.detection.taught_events import TaughtEvents
from emblema.evaluation.domain.exceptions import InvalidTaughtEventsError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from tests.evaluation.support import TASK, labelled

# Four events on the fitting side, one of them over three windows, among normal windows.
POOL = (
    labelled("a", 0, 10.0, 0.0),
    labelled("a", 1, 20.0, 1.0),
    labelled("a", 2, 30.0, 1.0),
    labelled("a", 3, 40.0, 1.0),
    labelled("b", 4, 10.0, 0.0),
    labelled("b", 5, 20.0, 2.0),
    labelled("c", 6, 10.0, 3.0),
    labelled("c", 7, 20.0, AnomalyScheme.NORMAL),
    labelled("d", 8, 10.0, 4.0),
)


def test_the_events_drawn_are_whole_and_their_windows_by_unit_and_place() -> None:
    taught = TaughtEvents.drawn(TASK, POOL, events=2, seed=1)

    assert taught.numbers() == (1, 2)
    assert [w for w in POOL if w.target in (1.0, 2.0)] == list(taught.windows)
    assert sum(w.target == 1.0 for w in taught.windows) == 3


def test_the_draws_nest_under_one_seed_and_differ_between_seeds() -> None:
    fewer = TaughtEvents.drawn(TASK, POOL, events=1, seed=3)
    more = TaughtEvents.drawn(TASK, POOL, events=3, seed=3)

    assert (fewer.numbers(), more.numbers()) == ((2,), (2, 3, 4))
    assert (
        len({TaughtEvents.drawn(TASK, POOL, events=1, seed=seed).numbers() for seed in range(8)})
        > 1
    )


def test_the_draw_does_not_depend_on_the_order_the_windows_arrive_in() -> None:
    forwards = TaughtEvents.drawn(TASK, POOL, events=2, seed=5)
    backwards = TaughtEvents.drawn(TASK, tuple(reversed(POOL)), events=2, seed=5)

    assert forwards == backwards


def test_every_event_is_drawn_when_as_many_are_asked_as_there_are() -> None:
    assert TaughtEvents.drawn(TASK, POOL, events=4, seed=1).numbers() == (1, 2, 3, 4)


def test_more_events_than_the_side_holds_or_none_at_all_are_refused() -> None:
    with pytest.raises(InvalidTaughtEventsError, match="holds 4 events"):
        TaughtEvents.drawn(TASK, POOL, events=5, seed=1)
    with pytest.raises(InvalidTaughtEventsError, match="at least one"):
        TaughtEvents.drawn(TASK, POOL, events=0, seed=1)


def test_windows_that_overlap_no_event_or_hold_another_count_are_refused() -> None:
    with pytest.raises(InvalidTaughtEventsError, match="overlaps an event"):
        TaughtEvents(task=TASK, windows=(labelled("a", 0, 10.0, 0.0),), events=1, seed=1)
    with pytest.raises(InvalidTaughtEventsError, match="were asked"):
        TaughtEvents(task=TASK, windows=(labelled("a", 1, 20.0, 1.0),), events=2, seed=1)
