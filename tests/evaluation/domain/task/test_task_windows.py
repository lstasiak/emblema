from emblema.evaluation.domain.task.task_windows import TaskWindows
from tests.evaluation.support import window

WINDOWS = (
    window("b", 3, 36.0),
    window("a", 1, 48.0),
    window("a", 0, 24.0),
    window("b", 2, 24.0),
    window("c", 4, 30.0),
)


def test_every_window_is_read_as_given() -> None:
    assert TaskWindows.EVERY.chosen_from(WINDOWS) == WINDOWS


def test_the_first_window_of_each_unit_is_the_earliest_whatever_order_it_comes_in() -> None:
    assert TaskWindows.FIRST.chosen_from(WINDOWS) == (
        window("a", 0, 24.0),
        window("b", 2, 24.0),
        window("c", 4, 30.0),
    )


def test_no_windows_leave_nothing_to_choose() -> None:
    assert TaskWindows.FIRST.chosen_from(()) == ()


def test_the_choice_reads_from_an_iterator_once() -> None:
    assert TaskWindows.FIRST.chosen_from(iter(WINDOWS[:2])) == WINDOWS[:2]
