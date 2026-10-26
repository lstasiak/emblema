"""The packet report finds the channels read at the same instants and compares them with the groups.

Its measures are held to hand-built instants whose answers are known, and the whole report to a
small mission written in the publisher's layout: a ``channels.csv`` and one zip per channel holding
a pickled data frame indexed by instants.
"""

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.esa_channel_packets import (
    NANOSECONDS,
    PacketRow,
    agreement,
    channel_row,
    components,
    measure,
    packet_rows,
    packets,
    pair_row,
    probe_windows,
    read_rows,
)

pd = pytest.importorskip("pandas")

START = 946_684_800 * NANOSECONDS
DESCRIPTION = {"Subsystem": "subsystem_1", "Group": "1", "Target": "YES"}


def seconds(*values: float) -> np.ndarray:
    return START + (np.asarray(values) * NANOSECONDS).astype(np.int64)


def test_a_channel_on_a_grid_has_one_step_and_the_bin_keeps_one_reading_in_three() -> None:
    times = seconds(*np.arange(0.0, 600.0, 10.0))

    row = channel_row("M", "c", DESCRIPTION, times, half=int(seconds(300.0)[0]))

    assert row.readings == 60
    assert row.step_median == row.step_p05 == row.step_p95 == 10.0
    assert row.steps_distinct == 1
    assert row.half_readings == 30
    assert row.half_kept == 10
    assert row.gap_share == 0.0


def test_a_long_silence_counts_as_a_gap_and_not_as_the_cadence() -> None:
    times = seconds(0.0, 1.0, 2.0, 3.0, 4.0, 100.0)

    row = channel_row("M", "c", DESCRIPTION, times, half=int(seconds(50.0)[0]))

    assert row.step_median == 1.0
    assert row.gap_share == pytest.approx(0.2)
    assert row.steps_distinct == 2


def test_channels_share_every_instant_none_or_meet_nowhere() -> None:
    grid = [seconds(0.0, 30.0, 60.0)]
    offset = [seconds(15.0, 45.0)]
    finer = [seconds(*np.arange(0.0, 61.0, 10.0))]
    nothing = [seconds()]

    assert pair_row("M", "a", "b", grid, grid).share == 1.0
    assert pair_row("M", "a", "b", grid, offset).share == 0.0
    assert pair_row("M", "a", "b", grid, finer).share == 1.0
    assert math.isnan(pair_row("M", "a", "b", grid, nothing).share)


def test_probe_windows_are_spread_over_the_span_and_stay_inside_it() -> None:
    windows = probe_windows(0, 100 * NANOSECONDS, 3, 10.0)

    assert windows == [
        (0, 10 * NANOSECONDS),
        (45 * NANOSECONDS, 55 * NANOSECONDS),
        (90 * NANOSECONDS, 100 * NANOSECONDS),
    ]


def test_parts_are_joined_through_any_pair_and_kept_in_order() -> None:
    parts = components(["a", "b", "c", "d"], [("c", "a"), ("d", "c")])

    assert parts == [["a", "c", "d"], ["b"]]


def write_mission(root: Path, channels: dict[str, tuple[str, np.ndarray]]) -> Path:
    mission = root / "ESA-Mission1"
    (mission / "channels").mkdir(parents=True)
    lines = ["Channel,Subsystem,Physical Unit,Group,Target,Categorical"]
    for name, (group, times) in channels.items():
        lines.append(f"{name},subsystem_1,physical_unit_1,{group},YES,NO")
        index = pd.DatetimeIndex(times.astype("datetime64[ns]"))
        frame = pd.DataFrame({name: np.zeros(times.size, dtype=np.float32)}, index=index)
        buffer = io.BytesIO()
        frame.to_pickle(buffer)
        with zipfile.ZipFile(mission / "channels" / f"{name}.zip", "w") as archive:
            archive.writestr(name, buffer.getvalue())
    (mission / "channels.csv").write_text("\n".join(lines) + "\n")
    return mission


@pytest.fixture
def reported(tmp_path: Path) -> Path:
    day = np.arange(0.0, 86_400.0, 30.0)
    mission = write_mission(
        tmp_path / "raw",
        {
            "channel_1": ("1", seconds(*day)),
            "channel_2": ("1", seconds(*day)),
            "channel_3": ("2", seconds(*(day + 15.0))),
        },
    )
    out = tmp_path / "out"
    measure(out, [mission], probes=4, seconds=3600.0)
    packets(out)
    return out


def test_channels_read_together_are_one_packet_and_match_the_declared_groups(
    reported: Path,
) -> None:
    rows = read_rows(reported / "packets.csv")
    channels = read_rows(reported / "channels.csv")

    assert [row["channels"] for row in rows] == ["channel_1 channel_2", "channel_3"]
    assert [row["groups"] for row in rows] == ["1", "2"]
    assert all(row["one_digest"] == "True" for row in rows)
    assert all(float(row["meets_others"]) == 0.0 for row in rows)
    assert len(read_rows(reported / "pairs.csv")) == 3
    assert {row["half_readings"] for row in channels} == {"1440"}


def test_a_packet_mixing_declared_groups_is_a_disagreement() -> None:
    channels = [
        {"mission": "M", "channel": "a", "group": "1"},
        {"mission": "M", "channel": "b", "group": "2"},
    ]
    mixed = PacketRow("M", 1, "a b", "1 2", True, 30.0, 10.0, 1.0, math.nan)

    [_, _, line] = agreement([mixed], channels)

    assert line == "| M | 2 | 1 | 2 | no | 1 |"


def test_a_pair_that_never_meets_joins_nothing() -> None:
    channels = [
        {
            "mission": "M",
            "channel": name,
            "group": "1",
            "digest": name,
            "step_median": "1.0",
            "readings": "10",
            "half_kept": "5",
            "half_readings": "5",
        }
        for name in ("a", "b")
    ]
    pairs = [{"mission": "M", "a": "a", "b": "b", "share": "nan"}]

    rows = packet_rows(channels, pairs, threshold=0.99)

    assert [row.channels for row in rows] == ["a", "b"]
    assert all(math.isnan(row.meets_others) for row in rows)
