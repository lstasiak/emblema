"""Group the raw channels of the ESA Anomaly Dataset by the instants they are sampled at.

A satellite sends its telemetry in packets: the channels of one packet are read at the same
instants, and different packets at different ones. The corpus reader keeps the earliest reading of
every 30-second bin, which lines all channels up on one grid and hides both the packets and the
uneven spacing inside a channel. This report measures them from the published archives.

    uv sync --all-extras
    uv run scripts/esa_channel_packets.py measure --out data/report/esa-channel-packets
    uv run scripts/esa_channel_packets.py packets --out data/report/esa-channel-packets

``measure`` reads every channel of each mission once and writes ``channels.csv`` (a channel's
declared group, its readings, its spacing and what the bin keeps of its first half) and
``pairs.csv`` (for every pair of a mission's channels, the share of instants they share inside
probe windows spread over the mission). ``packets`` reads those two files and joins the channels
whose shared share reaches the threshold into packets, writes ``packets.csv`` and prints how the
packets compare with the groups the publisher declares. The numbers describe the data and belong in
no result.
"""

import argparse
import csv
import hashlib
import math
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.raw_corpora import raw_root

MISSIONS = ("ESA-Mission1", "ESA-Mission2")
NANOSECONDS = 1_000_000_000
# The corpus reader's bin, stated here so that what it keeps is measured against the raw readings.
READER_BIN_SECONDS = 30
# A step this many times the channel's median step is a gap in the downlink, not its cadence.
GAP_FACTOR = 10


@dataclass(frozen=True)
class ChannelRow:
    """One channel's sampling; a plain record written to ``channels.csv``.

    Attributes:
        mission: The mission it belongs to.
        channel: Its name.
        subsystem: The subsystem the publisher assigns it to.
        group: The group the publisher assigns it to.
        target: Whether the benchmark scores anomalies on it.
        readings: Its readings over the whole mission.
        first: Its first instant, in nanoseconds since the epoch.
        last: Its last instant.
        half_readings: Its readings before the middle of the mission's span.
        half_kept: Of those, the readings the reader's bin keeps.
        step_median: The median spacing between its readings, in seconds.
        step_p05: The 5th percentile of that spacing.
        step_p95: The 95th percentile.
        steps_distinct: Distinct spacings, rounded to a tenth of a second.
        gap_share: Share of its steps longer than ``GAP_FACTOR`` times its median step.
        digest: A digest of all its instants: channels with the same one are read at the same
            instants over the whole mission.
    """

    mission: str
    channel: str
    subsystem: str
    group: str
    target: str
    readings: int
    first: int
    last: int
    half_readings: int
    half_kept: int
    step_median: float
    step_p05: float
    step_p95: float
    steps_distinct: int
    gap_share: float
    digest: str


@dataclass(frozen=True)
class PairRow:
    """How many instants two channels of one mission share; written to ``pairs.csv``.

    Attributes:
        mission: The mission.
        a: The first channel.
        b: The second channel.
        shared: Instants both read, summed over the probe windows both read in.
        sparser: The readings of the sparser of the two in each such window, summed.
        share: ``shared`` over ``sparser``; one where every reading of the sparser channel is
            also one of the other's, zero where they never meet. Not a number where the two
            never read in the same window.
    """

    mission: str
    a: str
    b: str
    shared: int
    sparser: int
    share: float


def declared(mission_root: Path) -> dict[str, dict[str, str]]:
    """The publisher's description of each channel, by channel name."""
    with (mission_root / "channels.csv").open(newline="") as file:
        return {row["Channel"]: row for row in csv.DictReader(file)}


def instants_of(archive: Path) -> NDArray[np.int64]:
    """The instants of one channel archive, in nanoseconds since the epoch, as published."""
    import pandas as pd

    with zipfile.ZipFile(archive) as opened:
        [member] = opened.namelist()
        with opened.open(member) as stream:
            frame = pd.read_pickle(stream)
    return np.asarray(frame.index.values.astype("datetime64[ns]").astype(np.int64))


def channel_row(
    mission: str, channel: str, description: Mapping[str, str], times: NDArray[np.int64], half: int
) -> ChannelRow:
    steps = np.diff(times) / NANOSECONDS
    before = times[times < half]
    bins = before // (READER_BIN_SECONDS * NANOSECONDS)
    median = float(np.median(steps)) if steps.size else float("nan")
    return ChannelRow(
        mission=mission,
        channel=channel,
        subsystem=description["Subsystem"],
        group=description["Group"],
        target=description["Target"],
        readings=int(times.size),
        first=int(times[0]),
        last=int(times[-1]),
        half_readings=int(before.size),
        half_kept=int(np.unique(bins).size),
        step_median=median,
        step_p05=float(np.percentile(steps, 5)) if steps.size else float("nan"),
        step_p95=float(np.percentile(steps, 95)) if steps.size else float("nan"),
        steps_distinct=int(np.unique(np.round(steps, 1)).size),
        gap_share=float((steps > GAP_FACTOR * median).mean()) if steps.size else float("nan"),
        digest=hashlib.sha256(times.tobytes()).hexdigest()[:16],
    )


def probe_windows(first: int, last: int, probes: int, seconds: float) -> list[tuple[int, int]]:
    """``probes`` windows of ``seconds`` each, spread evenly over ``[first, last]``."""
    width = int(seconds * NANOSECONDS)
    starts = np.linspace(first, max(first, last - width), probes)
    return [(int(start), int(start) + width) for start in starts]


def probed(times: NDArray[np.int64], windows: Sequence[tuple[int, int]]) -> list[NDArray[np.int64]]:
    """A channel's instants inside each window."""
    cuts = [np.searchsorted(times, edge) for window in windows for edge in window]
    return [times[cuts[2 * i] : cuts[2 * i + 1]] for i in range(len(windows))]


def pair_row(
    mission: str,
    a: str,
    b: str,
    first: Sequence[NDArray[np.int64]],
    second: Sequence[NDArray[np.int64]],
) -> PairRow:
    shared = sparser = 0
    for x, y in zip(first, second, strict=True):
        if x.size and y.size:
            shared += int(np.intersect1d(x, y, assume_unique=True).size)
            sparser += min(x.size, y.size)
    return PairRow(mission, a, b, shared, sparser, shared / sparser if sparser else float("nan"))


def measure_mission(
    mission_root: Path, probes: int, seconds: float
) -> tuple[list[ChannelRow], list[PairRow]]:
    """Every channel of one mission, read once each, and every pair of them."""
    mission = mission_root.name
    description = declared(mission_root)
    archives = {path.stem: path for path in (mission_root / "channels").glob("*.zip")}
    names = [name for name in description if name in archives]
    # The probe windows need the mission's span before any channel is probed, and a mission's
    # instants do not fit in memory at once, so the archives are read twice.
    spans = {}
    for name in names:
        times = instants_of(archives[name])
        spans[name] = (int(times[0]), int(times[-1]))
    first = min(start for start, _ in spans.values())
    last = max(end for _, end in spans.values())
    half = first + (last - first) // 2
    windows = probe_windows(first, last, probes, seconds)
    rows, inside = [], {}
    for name in names:
        times = instants_of(archives[name])
        rows.append(channel_row(mission, name, description[name], times, half))
        inside[name] = probed(times, windows)
        print(f"{mission}/{name}: {times.size:,} readings", flush=True)
    pairs = [pair_row(mission, a, b, inside[a], inside[b]) for a, b in combinations(names, 2)]
    return rows, pairs


def measure(out: Path, roots: Iterable[Path], probes: int = 24, seconds: float = 6 * 3600) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[ChannelRow] = []
    pairs: list[PairRow] = []
    for root in roots:
        held_rows, held_pairs = measure_mission(root, probes, seconds)
        rows.extend(held_rows)
        pairs.extend(held_pairs)
    write(out / "channels.csv", CHANNEL_COLUMNS, [asdict(row) for row in rows])
    write(out / "pairs.csv", PAIR_COLUMNS, [asdict(row) for row in pairs])


# --- packets -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketRow:
    """Channels of one mission read at the same instants; written to ``packets.csv``.

    Attributes:
        mission: The mission.
        packet: Its number within the mission, in the order of its first channel.
        channels: Its channels, space-separated.
        groups: The declared groups its channels belong to, space-separated.
        one_digest: Whether all its channels are read at the same instants over the whole
            mission, not only inside the probe windows.
        step_median: The median over its channels of their median step, in seconds.
        readings: The median over its channels of their readings.
        kept: The median over its channels of the share of their first half the bin keeps.
        meets_others: The largest share of instants it shares with a channel of another packet.
    """

    mission: str
    packet: int
    channels: str
    groups: str
    one_digest: bool
    step_median: float
    readings: float
    kept: float
    meets_others: float


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def components(names: Sequence[str], joined: Iterable[tuple[str, str]]) -> list[list[str]]:
    """The connected parts of the graph the pairs draw, in the order of their first name."""
    parent = {name: name for name in names}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b in joined:
        parent[root(a)] = root(b)
    parts: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parts[root(name)].append(name)
    return list(parts.values())


def packet_rows(
    channels: Sequence[Mapping[str, str]], pairs: Sequence[Mapping[str, str]], threshold: float
) -> list[PacketRow]:
    """The packets of each mission: channels joined where their shared share reaches threshold.

    A pair with no window in common joins nothing; its channels may still meet through others.
    """
    result = []
    for mission in dict.fromkeys(row["mission"] for row in channels):
        held = {row["channel"]: row for row in channels if row["mission"] == mission}
        shares = {
            (row["a"], row["b"]): float(row["share"]) for row in pairs if row["mission"] == mission
        }
        joined = [pair for pair, share in shares.items() if share >= threshold]
        parts = components(list(held), joined)
        packet_of = {name: number for number, part in enumerate(parts, 1) for name in part}
        for number, part in enumerate(parts, 1):
            rows = [held[name] for name in part]
            across = [
                share
                for (a, b), share in shares.items()
                if (packet_of[a] == number) != (packet_of[b] == number) and not math.isnan(share)
            ]
            result.append(
                PacketRow(
                    mission=mission,
                    packet=number,
                    channels=" ".join(part),
                    groups=" ".join(sorted({row["group"] for row in rows}, key=int)),
                    one_digest=len({row["digest"] for row in rows}) == 1,
                    step_median=float(np.median([float(row["step_median"]) for row in rows])),
                    readings=float(np.median([int(row["readings"]) for row in rows])),
                    kept=float(
                        np.median([int(r["half_kept"]) / int(r["half_readings"]) for r in rows])
                    ),
                    meets_others=max(across, default=float("nan")),
                )
            )
    return result


def agreement(packets: Sequence[PacketRow], channels: Sequence[Mapping[str, str]]) -> list[str]:
    """Per mission: packets, declared groups, and where the two partitions disagree."""
    lines = [
        "| Mission | Channels | Packets | Groups | Packets = groups | Packets of one digest |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for mission in dict.fromkeys(packet.mission for packet in packets):
        held = [p for p in packets if p.mission == mission]
        groups = {row["group"] for row in channels if row["mission"] == mission}
        same = all(len(p.groups.split()) == 1 for p in held) and len(held) == len(groups)
        lines.append(
            f"| {mission} | {sum(len(p.channels.split()) for p in held)} | {len(held)} | "
            f"{len(groups)} | {'yes' if same else 'no'} | "
            f"{sum(p.one_digest for p in held)} |"
        )
    return lines


def packet_table(packets: Sequence[PacketRow]) -> list[str]:
    lines = [
        "| Mission | Packet | Channels | Groups | Step s | Readings | Kept by bin | Meets others |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for p in packets:
        lines.append(
            f"| {p.mission} | {p.packet} | {len(p.channels.split())} | {p.groups} | "
            f"{p.step_median:.1f} | {p.readings:,.0f} | {p.kept:.1%} | {p.meets_others:.1%} |"
        )
    return lines


def packets(out: Path, threshold: float = 0.99) -> None:
    channels = read_rows(out / "channels.csv")
    rows = packet_rows(channels, read_rows(out / "pairs.csv"), threshold)
    write(out / "packets.csv", PACKET_COLUMNS, [asdict(row) for row in rows])
    for line in [*agreement(rows, channels), "", *packet_table(rows)]:
        print(line)


# --- common --------------------------------------------------------------------------------------

CHANNEL_COLUMNS = tuple(field.name for field in fields(ChannelRow))
PAIR_COLUMNS = tuple(field.name for field in fields(PairRow))
PACKET_COLUMNS = tuple(field.name for field in fields(PacketRow))


def write(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure", help="read every channel's instants")
    measuring.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/report/esa-channel-packets"
    )
    measuring.add_argument("--missions", nargs="+", default=list(MISSIONS))
    measuring.add_argument("--probes", type=int, default=24, help="probe windows per mission")
    measuring.add_argument("--hours", type=float, default=6.0, help="length of a probe window")
    grouping = commands.add_parser("packets", help="join the channels into packets")
    grouping.add_argument("--out", type=Path, default=REPO_ROOT / "data/report/esa-channel-packets")
    grouping.add_argument("--threshold", type=float, default=0.99)
    arguments = parser.parse_args()
    if arguments.command == "measure":
        root = raw_root("esa_ad")
        if root is None:
            parser.error("the ESA Anomaly Dataset is not downloaded under data/raw/esa_ad")
        roots = [root / mission for mission in arguments.missions]
        measure(arguments.out, roots, arguments.probes, arguments.hours * 3600)
    else:
        packets(arguments.out, arguments.threshold)


if __name__ == "__main__":
    main()
