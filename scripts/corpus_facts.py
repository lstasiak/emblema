"""Count the structural facts of each raw corpus and print them as TOML for the budget file.

Runs where the data is, never in CI. Per corpus it counts units, channels and observed values on the
pretraining side, and the windows and tokens of each window variant declared in
``corpus_budget.toml``, as a ``[corpora.<key>.measured]`` block to paste over the previous one.

    uv run scripts/corpus_facts.py                            # every corpus with data present
    uv run scripts/corpus_facts.py cmapss physionet2012
    uv run scripts/corpus_facts.py esa_ad          # needs the corpora extra for the pickles

A unit is what a split happens on (an engine, a run, a machine, an ICU stay, a mission); the
pretraining side is the official training portion; a token is one observed value; a window with
nothing observed is not a window; static descriptors count once per unit as timeless tokens.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# The arithmetic shared with the report lives in the sibling module; running this file as a script
# puts only ``scripts/`` on the path, so the repository root is added first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.corpus_budget_report import (
    DEFAULT_CONFIG,
    Budget,
    Corpus,
    Measured,
    MeasuredWindow,
    Subsampling,
    Window,
    windows_in_series,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

Measurer = Callable[[Path, Corpus, date], Measured]


def require(paths: Sequence[Path], what: str, root: Path) -> None:
    if not paths:
        raise SystemExit(f"no {what} under {root}; run fetch_corpora.py first")


def archive_bytes(root: Path) -> int:
    """Size of what was downloaded: the top-level files of the corpus directory.

    Markers and half-finished or rejected downloads are not part of the corpus.
    """
    return sum(
        path.stat().st_size
        for path in root.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix not in {".part", ".bad"}
    )


def summary(values: Sequence[int | float], label: str) -> str:
    return (
        f"{label} min/median/max {min(values):,.0f}/{statistics.median(values):,.0f}/"
        f"{max(values):,.0f}"
    )


def regular_windows(
    lengths: Sequence[int], channels: int, windows: Iterable[Window]
) -> tuple[MeasuredWindow, ...]:
    """Exact window and token counts for a regular corpus: every sample carries every channel."""
    result = []
    for window in windows:
        if window.unit != "samples":
            raise SystemExit(f"window {window.name}: a regular corpus is windowed in samples")
        count = sum(windows_in_series(length, window) for length in lengths)
        result.append(
            MeasuredWindow(
                name=window.name, count=count, tokens=count * int(window.length) * channels
            )
        )
    return tuple(result)


def timed_windows(
    series: Sequence[Sequence[float]], spans: Sequence[float], windows: Iterable[Window]
) -> tuple[MeasuredWindow, ...]:
    """Exact counts for an irregular corpus, windowed by time.

    ``series`` holds each unit's sorted observation times in hours from its start, ``spans`` the
    length of each unit in hours. A window covers ``[start, start + length)``; its tokens are the
    observations inside. A window that falls into a gap holds nothing to train on and is not
    counted.
    """
    result = []
    for window in windows:
        if window.unit != "hours":
            raise SystemExit(f"window {window.name}: an irregular corpus is windowed in hours")
        count = 0
        tokens = 0
        for times, span in zip(series, spans, strict=True):
            for index in range(windows_in_series(span, window)):
                start = index * window.stride
                inside = bisect.bisect_left(times, start + window.length) - bisect.bisect_left(
                    times, start
                )
                if inside:
                    count += 1
                    tokens += inside
        result.append(MeasuredWindow(name=window.name, count=count, tokens=tokens))
    return tuple(result)


def consistent(widths: set[int], what: str) -> int:
    if len(widths) != 1:
        raise SystemExit(f"inconsistent {what} across files: {sorted(widths)}")
    return widths.pop()


def measure_cmapss(root: Path, corpus: Corpus, today: date) -> Measured:
    files = sorted(root.rglob("train_FD00*.txt"))
    require(files, "train_FD00x.txt files", root)
    lengths: list[int] = []
    widths: set[int] = set()
    notes = []
    for file in files:
        cycles: Counter[int] = Counter()
        for line in file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                cycles[int(parts[0])] += 1
                widths.add(len(parts) - 5)  # unit, cycle and three operating settings
        lengths.extend(cycles.values())
        notes.append(f"{file.stem}: {len(cycles)} engines")
    channels = consistent(widths, "column counts")
    notes.append(summary(lengths, "cycles per engine"))
    notes.append("operating-setting columns are not counted as channels")
    return Measured(
        measured_on=today,
        units=len(lengths),
        channels=channels,
        observations=sum(lengths) * channels,
        archive_bytes=archive_bytes(root),
        windows=regular_windows(lengths, channels, corpus.windows),
        notes="; ".join(notes),
    )


def measure_skab(root: Path, corpus: Corpus, today: date) -> Measured:
    files = sorted(path for path in root.rglob("*.csv") if "data" in path.parts)
    require(files, "data/**/*.csv files", root)
    excluded = {"datetime", "anomaly", "changepoint"}
    lengths: list[int] = []
    spacings: list[float] = []
    widths: set[int] = set()
    folders: Counter[str] = Counter()
    for file in files:
        lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        header = lines[0].split(";")
        widths.add(len([column for column in header if column not in excluded]))
        lengths.append(len(lines) - 1)
        folders[file.parent.name] += 1
        first, second = (datetime.fromisoformat(line.split(";")[0]) for line in lines[1:3])
        spacings.append((second - first).total_seconds())
    channels = consistent(widths, "sensor columns")
    notes = [
        "files per folder "
        + ", ".join(f"{name}: {count}" for name, count in sorted(folders.items())),
        summary(lengths, "rows per file"),
        f"median spacing {statistics.median(spacings):.0f} s",
    ]
    return Measured(
        measured_on=today,
        units=len(lengths),
        channels=channels,
        observations=sum(lengths) * channels,
        archive_bytes=archive_bytes(root),
        windows=regular_windows(lengths, channels, corpus.windows),
        notes="; ".join(notes),
    )


def measure_smd(root: Path, corpus: Corpus, today: date) -> Measured:
    files = sorted(root.rglob("ServerMachineDataset/train/machine-*.txt"))
    require(files, "ServerMachineDataset/train/machine-*.txt files", root)
    lengths: list[int] = []
    widths: set[int] = set()
    for file in files:
        lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        lengths.append(len(lines))
        widths.add(len(lines[0].split(",")))
    channels = consistent(widths, "metric counts")
    groups = Counter(file.stem.split("-")[1] for file in files)
    notes = [
        f"{len(files)} machines in groups "
        + ", ".join(f"{g}: {n}" for g, n in sorted(groups.items())),
        summary(lengths, "rows per machine"),
        "one-minute sampling per the authors; test halves not counted",
    ]
    return Measured(
        measured_on=today,
        units=len(lengths),
        channels=channels,
        observations=sum(lengths) * channels,
        archive_bytes=archive_bytes(root),
        windows=regular_windows(lengths, channels, corpus.windows),
        notes="; ".join(notes),
    )


PHYSIONET_DESCRIPTORS = frozenset({"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"})
PHYSIONET_SPAN_HOURS = 48.0


def measure_physionet2012(root: Path, corpus: Corpus, today: date) -> Measured:
    files = sorted(root.rglob("set-a/*.txt"))
    require(files, "set-a/*.txt records", root)
    series: list[Sequence[float]] = []
    timeless: list[int] = []
    parameters: set[str] = set()
    for file in files:
        times: list[float] = []
        seen: set[str] = set()
        for line in file.read_text(encoding="utf-8").splitlines()[1:]:
            clock, parameter, _value = line.split(",")
            hour, minute = clock.split(":")
            at = int(hour) + int(minute) / 60
            # The general descriptors open every record at 00:00; a second value of the same
            # parameter (weight is also a time series) is an observation.
            if at == 0 and parameter in PHYSIONET_DESCRIPTORS and parameter not in seen:
                seen.add(parameter)
                continue
            times.append(at)
            parameters.add(parameter)
        times.sort()
        series.append(times)
        timeless.append(len(seen - {"RecordID"}))
    counts = [len(times) for times in series]
    other_sets = {name: len(list(root.rglob(f"{name}/*.txt"))) for name in ("set-b", "set-c")}
    notes = [
        "set-a stays counted; other sets present: "
        + ", ".join(f"{name}: {count}" for name, count in other_sets.items()),
        summary(counts, "observations per stay"),
        f"latest observation at {max(max(times) for times in series if times):.1f} h",
        f"{len(parameters)} distinct time-series parameters",
    ]
    return Measured(
        measured_on=today,
        units=len(series),
        channels=len(parameters),
        observations=sum(counts),
        archive_bytes=archive_bytes(root),
        timeless_tokens_per_unit=round(statistics.mean(timeless)),
        windows=timed_windows(series, [PHYSIONET_SPAN_HOURS] * len(series), corpus.windows),
        notes="; ".join(notes),
    )


def channel_number(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def select_channels(
    policy: Subsampling, mission: str, names: Sequence[str], targets: Sequence[bool]
) -> list[str]:
    if policy.channel_set == "all":
        return list(names)
    if policy.channel_set == "lightweight" and mission in policy.lightweight:
        low, high = policy.lightweight[mission]
        return [name for name in names if low <= channel_number(name) <= high]
    return [name for name, target in zip(names, targets, strict=True) if target]


def read_channel_table(path: Path) -> tuple[list[str], list[bool]]:
    """Channel names and target flags from ``channels.csv``; without a target column all are."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit(f"{path} is empty")
    columns = list(rows[0])
    name_key = next((column for column in columns if "channel" in column.lower()), None)
    if name_key is None:
        raise SystemExit(f"{path}: no column names the channel; columns are {columns}")
    target_key = next((column for column in columns if "target" in column.lower()), None)
    names = [row[name_key].strip() for row in rows]
    truthy = {"1", "true", "yes", "target"}
    targets = [target_key is None or str(row[target_key]).strip().lower() in truthy for row in rows]
    return names, targets


@dataclass(frozen=True)
class DecimatedChannel:
    """One channel reduced to its occupied time bins.

    Attributes:
        first: Earliest timestamp, seconds since the epoch.
        last: Latest timestamp, seconds since the epoch.
        bins: Sorted, distinct bin indices (timestamp // spacing) that hold an observation.
        counts: Native observations per bin, aligned with ``bins``.
        median_spacing: Median gap between consecutive native observations, seconds.
    """

    first: int
    last: int
    bins: NDArray[np.int64]
    counts: NDArray[np.int64]
    median_spacing: float


def decimate(seconds: NDArray[np.int64], spacing: int) -> DecimatedChannel:
    """Reduce sorted timestamps to at most one observation per bin of ``spacing`` seconds.

    The raw array can be dropped afterwards: everything the counting needs — where the channel
    starts and ends, which bins are occupied and how densely — is kept, at a fraction of the size.
    """
    if len(seconds) == 0:
        raise SystemExit("a channel without observations cannot be decimated")
    bins, counts = np.unique(seconds // spacing, return_counts=True)
    median = float(np.median(np.diff(seconds))) if len(seconds) > 1 else math.nan
    return DecimatedChannel(int(seconds[0]), int(seconds[-1]), bins, counts, median)


def channel_seconds(path: Path) -> NDArray[np.int64]:
    """Observation times of one satellite channel, in seconds since the epoch.

    A channel is a pandas DataFrame pickled inside a zip; the times are its index, or its first
    column where the pickle did not keep one.
    """
    # Imported here: the extra that provides pandas is needed for this corpus alone.
    import pandas as pd

    frame = pd.read_pickle(path)
    index = frame.index if isinstance(frame.index, pd.DatetimeIndex) else frame.iloc[:, 0]
    return np.asarray(pd.to_datetime(index).values, dtype="datetime64[s]").astype(np.int64)


def measure_esa_ad(root: Path, corpus: Corpus, today: date) -> Measured:
    policy = corpus.subsampling
    if policy is None:
        raise SystemExit("esa_ad needs a [corpora.esa_ad.subsampling] table")
    missions = sorted(
        path
        for path in root.rglob("ESA-Mission*")
        if path.is_dir() and (path / "channels").is_dir()
    )
    require(missions, "ESA-Mission*/channels directories", root)
    spacing = int(policy.min_spacing_seconds)
    series: list[Sequence[float]] = []
    spans: list[float] = []
    counted = 0
    selected_total = 0
    kept_total = 0
    native_total = 0
    notes = []
    for mission in missions:
        names, targets = read_channel_table(mission / "channels.csv")
        selected = select_channels(policy, mission.name, names, targets)
        if not selected:
            raise SystemExit(f"{mission.name}: the {policy.channel_set} policy selects no channel")
        channels = [
            decimate(np.sort(channel_seconds(mission / "channels" / f"{name}.zip")), spacing)
            for name in selected
        ]
        start = min(channel.first for channel in channels)
        end = max(channel.last for channel in channels)
        cutoff = start + (end - start) // 2 if policy.portion == "first-half" else end + 1
        # The training portion ends at a bin boundary, at most one bin away from the exact half.
        cutoff_bin = cutoff // spacing
        native = 0
        kept_bins = []
        for channel in channels:
            training = channel.bins < cutoff_bin
            native += int(channel.counts[training].sum())
            kept_bins.append(channel.bins[training])
        kept = sum(len(bins) for bins in kept_bins)
        # Times in hours from the mission start, one entry per kept observation; a compact array
        # because a mission holds tens of millions of them.
        times = array("d")
        times.frombytes(
            ((np.sort(np.concatenate(kept_bins)) - start // spacing) * (spacing / 3600)).tobytes()
        )
        span_hours = (cutoff - start) / 3600
        included = policy.missions is None or mission.name in policy.missions
        if included:
            series.append(times)
            spans.append(span_hours)
            counted += 1
            selected_total += len(selected)
            kept_total += kept
            native_total += native
        spacings = [c.median_spacing for c in channels if not math.isnan(c.median_spacing)]
        spacing_note = (
            f"median native spacing {statistics.median(spacings):.0f} s"
            if spacings
            else "no spacing measurable"
        )
        notes.append(
            f"{mission.name}{'' if included else ' (measured, not counted)'}: {len(names)} "
            f"channels, {len(selected)} selected ({policy.channel_set}), span "
            f"{(end - start) / 3600 / 8766:.1f} y, training {span_hours / 8766:.1f} y, native "
            f"{native:,} -> {kept:,} at <= 1 per {spacing} s, {spacing_note}"
        )
    return Measured(
        measured_on=today,
        units=counted,
        channels=selected_total,
        observations=kept_total,
        native_observations=native_total,
        archive_bytes=archive_bytes(root),
        windows=timed_windows(series, spans, corpus.windows),
        notes="; ".join(notes),
    )


MEASURERS: dict[str, Measurer] = {
    "cmapss": measure_cmapss,
    "skab": measure_skab,
    "smd": measure_smd,
    "physionet2012": measure_physionet2012,
    "esa_ad": measure_esa_ad,
}


def render(key: str, measured: Measured) -> str:
    lines = [
        f"[corpora.{key}.measured]",
        f"measured_on = {measured.measured_on.isoformat()}",
        f"units = {measured.units}",
        f"channels = {measured.channels}",
        f"observations = {measured.observations}",
    ]
    if measured.native_observations is not None:
        lines.append(f"native_observations = {measured.native_observations}")
    if measured.archive_bytes is not None:
        lines.append(f"archive_bytes = {measured.archive_bytes}")
    lines += [
        f"timeless_tokens_per_unit = {measured.timeless_tokens_per_unit}",
        f"notes = {json.dumps(measured.notes)}",
    ]
    for window in measured.windows:
        lines += [
            "",
            f"[[corpora.{key}.measured.windows]]",
            f'name = "{window.name}"',
            f"count = {window.count}",
            f"tokens = {window.tokens}",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpora", nargs="*", help="corpus keys; default: every measurable corpus")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    budget = Budget.load(args.config)
    measurable = [key for key in budget.corpora if key in MEASURERS]
    keys = args.corpora or measurable
    unknown = sorted(set(keys) - set(measurable))
    if unknown:
        raise SystemExit(
            f"cannot measure {', '.join(unknown)}; measurable: {', '.join(measurable)}"
        )
    today = date.today()
    for key in keys:
        root = args.data_dir / key
        if not root.exists():
            sys.stderr.write(f"{key}: nothing under {root}; skipped\n")
            continue
        print(render(key, MEASURERS[key](root, budget.corpora[key], today)))
        print()


if __name__ == "__main__":
    main()
