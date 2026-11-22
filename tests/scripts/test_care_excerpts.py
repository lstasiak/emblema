"""The fault benchmark's excerpts: which turbines are read and which are never opened."""

from pathlib import Path

import pytest

from scripts.care_excerpts import (
    Event,
    HeldUnreadError,
    events_of,
    file_of,
    held_unread,
    read_events,
)

FARM_C = (2, 5, 12, 13, 14, 15, 16, 21, 23, 32, 33, 34, 35, 38, 42, 43, 44, 50, 52, 53, 55, 56)


def benchmark(root: Path) -> Path:
    """A farm B of nine turbines, one event each, and a farm A spelling its turbine's column."""
    for farm, column, turbines in (("A", "asset", (0, 10)), ("B", "asset_id", range(9))):
        folder = root / f"Wind Farm {farm}"
        (folder / "datasets").mkdir(parents=True)
        lines = [f"{column};event_id;event_label;event_start;event_start_id;event_end;event_end_id"]
        for turbine in turbines:
            label = "anomaly" if turbine % 2 else "normal"
            starts = f"2023-01-0{1 + turbine % 9} 00:00:00"
            lines.append(
                f"{turbine};{10 + turbine};{label};{starts};{100 + turbine};"
                f"2023-02-01 00:00:00;{900 + turbine}"
            )
        (folder / "event_info.csv").write_text("\n".join(lines) + "\n")
    return root


def test_the_turbines_held_unread_are_the_registered_ones() -> None:
    assert held_unread("A", (0, 10, 11, 13, 21)) == {0, 21}
    assert held_unread("B", (0, 2, 5, 6, 7, 11, 12, 13, 14)) == {0, 5, 6, 14}
    assert held_unread("C", FARM_C) == {13, 21, 23, 32, 33, 38, 42, 50, 52, 53, 56}


def test_a_farms_events_are_read_under_either_spelling_of_the_turbines_column(
    tmp_path: Path,
) -> None:
    root = benchmark(tmp_path)

    assert events_of(root, "A")[1] == Event(
        farm="A",
        event=20,
        turbine=10,
        anomaly=False,
        first_row=110,
        last_row=910,
        starts="2023-01-02 00:00:00",
        ends="2023-02-01 00:00:00",
    )
    assert [event.anomaly for event in events_of(root, "B")][:3] == [False, True, False]


def test_the_events_read_leave_out_the_turbines_held_unread(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    held = held_unread("B", range(9))

    read = read_events(root, farms=("B",))

    assert len(held) == 4
    assert {event.turbine for event in read} == set(range(9)) - held


def test_a_file_of_a_turbine_held_unread_is_refused(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    held = next(
        event for event in events_of(root, "B") if event.turbine in held_unread("B", range(9))
    )

    with pytest.raises(HeldUnreadError, match="is held unread"):
        file_of(root, held)


def test_a_read_turbines_file_is_named_by_its_event(tmp_path: Path) -> None:
    root = benchmark(tmp_path)
    event = read_events(root, farms=("B",))[0]

    assert file_of(root, event) == root / "Wind Farm B" / "datasets" / f"{event.event}.csv"


def test_the_excerpt_lying_inside_a_fault_is_left_out_of_rankings() -> None:
    inside = Event("B", 87, 7, False, 52475, 54779, "2023-09-14", "2023-09-30")
    other = Event("B", 27, 7, True, 52619, 61403, "2023-09-01", "2023-11-01")

    assert not inside.ranked
    assert other.ranked
