"""The fault benchmark's excerpts as the detection reading takes them from their files.

CARE to Compare (Zenodo 15846963, version 6) is 95 excerpts of 36 turbines in three farms: a
year of ten-minute rows to fit on and a scored window after it, holding one annotated event, the
lead-up to a fault or a stretch without one. The benchmark joins no corpus (its excerpts of one
turbine hold the same rows, and its time is anonymised), so the reading takes it from the files.

Half of each farm's turbines are held unread, the first half in the project's seeded order, as
the two farms' turbines were held; this module names them and refuses their files, so that no
script of the reading can open one by an argument's mistake.

An excerpt is read on the axis of its clock hours. An hour is whole when it holds its six rows
and all are of normal operation (status 0 or 2); the detectors read whole hours only, so the
count of a channel's readings in an hour is the same everywhere and says nothing of the status.
A reading is missing where the file holds none and in a row with half its averages or more at
exactly nought, which is how two farms' loggers wrote a dropped row; a channel is read in an
hour only when all six of its readings are.
"""

from __future__ import annotations

import csv
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.domain.channels.channel_statistics import MAD_TO_DEVIATION  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from scripts.wind_fleet_audit import BOUND, CELSIUS, held_out  # noqa: E402

# pandas ships no type information, so a frame is whatever it hands back.
Frame = Any
Floats = NDArray[np.float64]
Bools = NDArray[np.bool_]
Ints = NDArray[np.int64]

FARMS = ("A", "B", "C")
SEED = 1
ANOMALY = "anomaly"
# An excerpt without a fault whose event lies inside the lead-up to a fault of the same turbine
# (excerpt 27), in the same rows: a detector that flags it is right, so no ranking of excerpts
# by their label reads it.
INSIDE_A_FAULT = frozenset({("B", 87)})
TRAINING = "train"
NORMAL_STATUSES = (0, 2)
ROWS_PER_HOUR = 6
# The share of a row's averages at exactly nought from which the row is a logger's dropped one.
DROPPED_ROW = 0.5
AVERAGE = "_avg"
# What a normal-behaviour model knows of an hour, per farm, by the benchmark's descriptions: the
# wind speed, the active power, the rotor's speed, the blades' pitch and the temperatures
# outside. The component temperatures are every other sensor in degrees Celsius.
CONDITIONS = {
    "A": ("wind_speed_3", "power_30", "sensor_52", "sensor_5", "sensor_0"),
    "B": ("wind_speed_61", "power_62", "sensor_25", "sensor_10", "sensor_8"),
    "C": (
        "wind_speed_236",
        "power_6",
        "sensor_144",
        "sensor_103",
        "sensor_7",
        "sensor_41",
        "sensor_177",
    ),
}
# Where the active power stands among a farm's conditions.
POWER = 1


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
    """

    farm: str
    event: int
    turbine: int
    anomaly: bool
    first_row: int
    last_row: int

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


def chosen(
    events: Sequence[Event], farms: Sequence[str], named: Sequence[Sequence[str]]
) -> list[Event]:
    """The excerpts a run reads: the farms', or among them the ones named.

    Raises:
        ValueError: If an excerpt named is not among the farms' excerpts that are read.
    """
    of_farms = [event for event in events if event.farm in farms]
    if not named:
        return of_farms
    wanted = {(farm, int(number)) for farm, number in named}
    if unknown := wanted - {(event.farm, event.event) for event in of_farms}:
        raise ValueError(
            "not among the excerpts read: "
            + ", ".join(f"{farm} {number}" for farm, number in sorted(unknown))
        )
    return [event for event in of_farms if (event.farm, event.event) in wanted]


def file_of(root: Path, event: Event) -> Path:
    """The excerpt's file.

    The turbine is taken from the farm's table, not from the event handed in, so an event built
    by hand cannot name a held turbine's excerpt as another's.

    Raises:
        ValueError: If the farm's table holds no such excerpt of that turbine.
        HeldUnreadError: If the excerpt's turbine is held unread.
    """
    turbines = {other.event: other.turbine for other in events_of(root, event.farm)}
    if turbines.get(event.event) != event.turbine:
        raise ValueError(
            f"farm {event.farm}'s table holds no excerpt {event.event} of turbine {event.turbine}"
        )
    if event.turbine in held_unread(event.farm, turbines.values()):
        raise HeldUnreadError(
            f"turbine {event.turbine} of farm {event.farm} is held unread; excerpt "
            f"{event.event} is not opened"
        )
    return folder_of(root, event.farm) / "datasets" / f"{event.event}.csv"


@dataclass(frozen=True)
class Sensors:
    """A farm's sensors as the reading groups them, each named by its column of averages.

    Attributes:
        channels: What the channel-local detector reads: every average that is neither an angle,
            which wraps, nor a counter, which only grows.
        temperatures: The component temperatures a normal-behaviour model answers.
        conditions: What that model knows of the hour.
    """

    channels: tuple[str, ...]
    temperatures: tuple[str, ...]
    conditions: tuple[str, ...]

    @classmethod
    def described(
        cls, farm: str, rows: Iterable[dict[str, str]], averages: Sequence[str]
    ) -> Sensors:
        """The groups from a farm's table of descriptions and the averages its files hold.

        Raises:
            ValueError: If a condition of the farm has no column of averages.
        """
        held = set(averages)
        conditions = tuple(f"{name}{AVERAGE}" for name in CONDITIONS[farm])
        if missing := [name for name in conditions if name not in held]:
            raise ValueError(f"farm {farm} holds no average of {', '.join(missing)}")
        channels: list[str] = []
        temperatures: list[str] = []
        for row in rows:
            name = f"{row['sensor_name']}{AVERAGE}"
            if name not in held:
                continue
            if row["is_angle"] != "True" and row["is_counter"] != "True":
                channels.append(name)
            if row["unit"] in CELSIUS and name not in conditions:
                temperatures.append(name)
        return cls(tuple(channels), tuple(temperatures), conditions)


def sensors_of(root: Path, farm: str) -> Sensors:  # pragma: no cover - reads the benchmark
    with (folder_of(root, farm) / "feature_description.csv").open(
        encoding="utf-8", errors="replace", newline=""
    ) as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    read = next(event for event in read_events(root, (farm,)))
    with file_of(root, read).open(encoding="utf-8", errors="replace", newline="") as source:
        header = next(csv.reader(source, delimiter=";"))
    return Sensors.described(farm, rows, [name for name in header if name.endswith(AVERAGE)])


@dataclass(frozen=True)
class Excerpt:
    """One excerpt's rows in time order.

    Attributes:
        event: The excerpt's event.
        columns: The averages read, in the order of ``values``.
        stamps: When each row was written, to the second.
        ids: Each row's number in the file, by which the event is placed.
        fitted: Whether a row belongs to the year a detector is fitted on.
        normal: Whether a row is of normal operation.
        values: ``[rows, columns]``, missing where a reading is.
    """

    event: Event
    columns: tuple[str, ...]
    stamps: NDArray[np.datetime64]
    ids: Ints
    fitted: Bools
    normal: Bools
    values: Floats

    @classmethod
    def of(
        cls,
        event: Event,
        columns: Sequence[str],
        *,
        stamps: NDArray[np.datetime64],
        ids: Ints,
        training: Sequence[str],
        status: Ints,
        values: Floats,
    ) -> Excerpt:
        """An excerpt from its file's columns, the dropped rows' readings made missing.

        Raises:
            ValueError: If the rows are not in time order, or the fitted year does not come
                first.
        """
        if len(stamps) > 1 and not (np.diff(stamps.astype(np.int64)) > 0).all():
            raise ValueError(f"excerpt {event.event} of farm {event.farm} is not in time order")
        fitted = np.asarray([part == TRAINING for part in training], dtype=np.bool_)
        if (np.diff(fitted.astype(np.int8)) > 0).any():
            raise ValueError(
                f"excerpt {event.event} of farm {event.farm} holds fitted rows after scored ones"
            )
        dropped = (values == 0.0).mean(axis=1) >= DROPPED_ROW
        return cls(
            event=event,
            columns=tuple(columns),
            stamps=stamps,
            ids=ids,
            fitted=fitted,
            normal=np.isin(status, NORMAL_STATUSES),
            values=np.where(dropped[:, None], np.nan, values),
        )

    def columns_at(self, names: Sequence[str]) -> Ints:
        """Where the named averages stand in ``values``."""
        place = {name: index for index, name in enumerate(self.columns)}
        return np.asarray([place[name] for name in names], dtype=np.int64)


def read_excerpt(root: Path, event: Event) -> Excerpt:  # pragma: no cover - reads the benchmark
    """An excerpt with every average its file holds."""
    import pandas as pd

    path = file_of(root, event)
    header: Frame = pd.read_csv(path, sep=";", nrows=0).columns
    averages = [name for name in header if name.endswith(AVERAGE)]
    frame: Frame = pd.read_csv(
        path,
        sep=";",
        usecols=["time_stamp", "id", "train_test", "status_type_id", *averages],
        parse_dates=["time_stamp"],
    )
    return Excerpt.of(
        event,
        averages,
        stamps=frame.time_stamp.to_numpy().astype("datetime64[s]"),
        ids=frame.id.to_numpy(dtype=np.int64),
        training=frame.train_test.tolist(),
        status=frame.status_type_id.to_numpy(dtype=np.int64),
        values=frame[averages].to_numpy(dtype=np.float64),
    )


@dataclass(frozen=True)
class Hours:
    """An excerpt on the axis of its clock hours, every hour from its first row's to its last's.

    Attributes:
        of_row: Each row's hour.
        slot: Each row's place among its hour's six.
        count: How many hours the axis holds.
        fitted: How many hours, from the first, hold no scored row.
        whole: Per hour, whether it holds six rows, all of normal operation.
        first: The hour of the event's first row.
        last: The hour of its last row.
    """

    of_row: Ints
    slot: Ints
    count: int
    fitted: int
    whole: Bools
    first: int
    last: int

    @classmethod
    def of(cls, excerpt: Excerpt) -> Hours:
        """The axis of an excerpt.

        Raises:
            ValueError: If the event's rows are not in the excerpt, or start inside the fitted
                year.
        """
        seconds = excerpt.stamps.astype("datetime64[s]").astype(np.int64)
        hours = seconds // 3600
        of_row = hours - hours[0]
        slot = (seconds % 3600) // 600
        count = int(of_row[-1]) + 1
        rows = np.bincount(of_row, minlength=count)
        normal = np.bincount(of_row, weights=excerpt.normal, minlength=count)
        scored = of_row[~excerpt.fitted]
        inside = (excerpt.ids >= excerpt.event.first_row) & (excerpt.ids <= excerpt.event.last_row)
        if not inside.any():
            raise ValueError(
                f"excerpt {excerpt.event.event} of farm {excerpt.event.farm} holds no row of its "
                "event"
            )
        fitted = int(scored[0]) if len(scored) else count
        first, last = int(of_row[inside][0]), int(of_row[inside][-1])
        if first < fitted:
            raise ValueError(
                f"the event of excerpt {excerpt.event.event} of farm {excerpt.event.farm} starts "
                "inside the fitted year"
            )
        return cls(
            of_row=of_row,
            slot=slot,
            count=count,
            fitted=fitted,
            whole=(rows == ROWS_PER_HOUR) & (normal == ROWS_PER_HOUR),
            first=first,
            last=last,
        )

    def readings(self, values: Floats) -> Floats:
        """``[hours, columns, 6]``: each whole hour's six readings of each column.

        Missing in an hour that is not whole, and for a column that lacks one of its six.
        """
        laid = np.full((self.count, ROWS_PER_HOUR, values.shape[1]), np.nan)
        laid[self.of_row, self.slot] = values
        laid[~self.whole] = np.nan
        lacking = np.isnan(laid).any(axis=1, keepdims=True)
        return np.where(lacking, np.nan, laid).transpose(0, 2, 1)

    def abnormal_share(self, normal: Bools) -> Floats:
        """Per hour, the share of its rows not of normal operation; one where it holds no row."""
        rows = np.bincount(self.of_row, minlength=self.count)
        held = np.bincount(self.of_row, weights=normal, minlength=self.count)
        with np.errstate(invalid="ignore", divide="ignore"):
            share: Floats = np.where(rows > 0, 1.0 - held / rows, 1.0)
        return share


def robust_scores(readings: Floats, fitted: int) -> Floats:
    """The readings as the publications score a channel.

    From the channel's median over the fitted hours, in robust deviations, held to the bound;
    in standard deviations where most of its readings agree, and as they are where all do.
    """
    channels = readings.shape[1]
    year: Floats = readings[:fitted].transpose(1, 0, 2).reshape(channels, -1)
    # A channel with no reading in the fitted hours has no median; it stays missing, silently.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(year, axis=1)
        spread = MAD_TO_DEVIATION * np.nanmedian(np.abs(year - centre[:, None]), axis=1)
        deviation = np.nanstd(year, axis=1)
    scale = np.where(spread > 0.0, spread, np.where(deviation > 0.0, deviation, 1.0))
    scored: Floats = np.clip(
        (readings - centre[None, :, None]) / scale[None, :, None], -BOUND, BOUND
    )
    return scored


def cells_of(readings: Floats, chunk: int = 1024) -> Iterator[Floats]:
    """The readings as the cells a detector over binned windows is laid from.

    ``[hours, channels, 6, features]`` in chunks: each reading a span of its own, so a span's
    mean, least, largest and last are the reading and its count is one.
    """
    names = BinnedTokens.FEATURE_NAMES
    values = [names.index(name) for name in ("mean", "minimum", "maximum", "first", "last")]
    for start in range(0, len(readings), chunk):
        part = readings[start : start + chunk]
        present = ~np.isnan(part)
        cells = np.zeros((*part.shape, len(names)))
        cells[..., BinnedTokens.PRESENCE] = present
        cells[..., BinnedTokens.LOG_COUNT] = np.where(present, np.log1p(1.0), 0.0)
        cells[..., values] = np.where(present, part, 0.0)[..., None]
        yield cells
