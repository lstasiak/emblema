"""The fault benchmark's excerpts as the detection reading takes them from their files.

CARE to Compare (Zenodo 15846963, version 6) is 95 excerpts of 36 turbines in three farms: a
year of ten-minute rows to fit on and a scored window after it, holding one annotated event, the
lead-up to a fault or a stretch without one. The benchmark joins no corpus (its excerpts of one
turbine hold the same rows, and its time is anonymised), so the reading takes it from the files.

Half of each farm's turbines are held unread, the first half in the project's seeded order, as
the two farms' turbines were held; this module names them and refuses their files, so that no
script of the reading can open one by an argument's mistake.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.wind_fleet_audit import held_out  # noqa: E402

FARMS = ("A", "B", "C")
SEED = 1
ANOMALY = "anomaly"
# An excerpt without a fault whose event lies inside the lead-up to a fault of the same turbine,
# in the same rows, and the excerpt of that fault: a detector that flags it is right, so no
# ranking of excerpts by their label reads it.
INSIDE_A_FAULT = {("B", 87): 27}


class HeldUnreadError(ValueError):
    """A file of a turbine held unread was asked for."""


@dataclass(frozen=True)
class Event:
    """One excerpt's annotated event, from its farm's table of events.

    Attributes:
        farm: The farm's letter.
        event: The excerpt's number, which names its file.
        turbine: The turbine the excerpt was cut from.
        anomaly: Whether the event is the lead-up to a fault.
        first_row: The ``id`` of the event's first row in the excerpt's file.
        last_row: The ``id`` of its last row.
        starts: When the event starts, as the table spells it.
        ends: When it ends.
    """

    farm: str
    event: int
    turbine: int
    anomaly: bool
    first_row: int
    last_row: int
    starts: str
    ends: str

    @property
    def ranked(self) -> bool:
        """Whether a ranking of excerpts by their label reads this one."""
        return (self.farm, self.event) not in INSIDE_A_FAULT


def folder_of(root: Path, farm: str) -> Path:
    return root / f"Wind Farm {farm}"


def events_of(root: Path, farm: str) -> list[Event]:
    """Every event of a farm, the held turbines' among them: the table is the farm's."""
    with (folder_of(root, farm) / "event_info.csv").open(
        encoding="utf-8", errors="replace", newline=""
    ) as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    return [
        Event(
            farm=farm,
            event=int(row["event_id"]),
            # The first farm's table names the turbine's column differently.
            turbine=int(row["asset_id"] if "asset_id" in row else row["asset"]),
            anomaly=row["event_label"] == ANOMALY,
            first_row=int(row["event_start_id"]),
            last_row=int(row["event_end_id"]),
            starts=row["event_start"],
            ends=row["event_end"],
        )
        for row in rows
    ]


def held_unread(farm: str, turbines: Iterable[int]) -> frozenset[int]:
    """The turbines of a farm no reading opens: the first half in the seeded order."""
    names = sorted({str(turbine) for turbine in turbines})
    return frozenset(int(name) for name in held_out(farm, names, len(names) // 2, SEED))


def read_events(root: Path, farms: Sequence[str] = FARMS) -> list[Event]:
    """The events of the turbines that are read, farm after farm in the tables' order."""
    kept: list[Event] = []
    for farm in farms:
        events = events_of(root, farm)
        held = held_unread(farm, (event.turbine for event in events))
        kept += [event for event in events if event.turbine not in held]
    return kept


def file_of(root: Path, event: Event) -> Path:
    """The excerpt's file.

    Raises:
        HeldUnreadError: If the excerpt's turbine is held unread.
    """
    held = held_unread(event.farm, (other.turbine for other in events_of(root, event.farm)))
    if event.turbine in held:
        raise HeldUnreadError(
            f"turbine {event.turbine} of farm {event.farm} is held unread; excerpt "
            f"{event.event} is not opened"
        )
    return folder_of(root, event.farm) / "datasets" / f"{event.event}.csv"
