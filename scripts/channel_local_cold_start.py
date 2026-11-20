"""The channel-local detector fitted on a mission's first days, read on the validation months.

Each departure reads a channel's hour against that channel's own recent hours, so it needs no
history beyond them. What the detector fits is the floor under each channel's spread and the
level of its alarm, both from its fitting side. This report fits them on the windows of the first
mission's first N days instead of its whole fitting side, and reads the validation months as the
verdict reads them:

- for every departure arm, the events found with 1 % of normal windows flagged, less those found
  when every month's scores are turned;
- at the detector's own alarm, the satellite benchmark's corrected event-wise F0.5, the share of
  normal windows flagged and the runs of false alarms.

A fitting window is read only with its widest context inside the mission, as a validation window
is only with its context inside the months loaded. Each reading comes beside the same detector
fitted on the whole fitting side so read, paired over draws of months. The CSV comes first, the
tables from it.

    uv run --env-file .env.r2 scripts/channel_local_cold_start.py windows --task NAME \
        --task-id UUID --manifest KEY CHECKSUM --weights NAME KEY CHECKSUM --out WINDOWS.csv
    uv run scripts/channel_local_cold_start.py curve --run RUN --cells CELLS.npy \
        --windows WINDOWS.csv --days 2 7 14 30 60 90 180 365 --out OUT
    uv run scripts/channel_local_cold_start.py tables --out OUT
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.detection.robust_scale import RobustScale  # noqa: E402
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm  # noqa: E402
from scripts.reporting import table  # noqa: E402
from scripts.telemetry_channel_local_report import (  # noqa: E402
    LOCAL_DETECTORS,
    Arm,
    SideArrays,
    arms_of,
    bin_values,
    departures_from,
    global_floor,
    window_scores,
)
from scripts.telemetry_context_report import REFIT, read_truncated  # noqa: E402
from scripts.telemetry_verdict_measure_report import (  # noqa: E402
    BETA,
    Tally,
    alarms,
    chance_by_month,
    month_draws,
    tally,
)

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

HOURS_PER_DAY = 24
# The widest span of the departure arms: a window is read only with this many hours before it.
WIDEST_HOURS = 72
# The arm the verdict's F-score bar chose on the whole fitting side, followed along the days.
F_BAR = ("max", "bins-72h")
# The share of fitting windows the alarm is set to flag, as the verdict's arms set it.
CONTAMINATION = 0.02
ROTATIONS = 1000
RESAMPLES = 2000
SEED = 1
ALL = "all"
FEATURES = ("hour", "bins")
SPANS = (6, 24, 72)
CHUNK = 1024
WINDOW_COLUMNS = ("index", "side", "unit", "position", "ends_at", "target")
CURVE_COLUMNS = (
    "days",
    "detector",
    "arm",
    "fit_windows",
    "found_at_1",
    "excess_at_1",
    "excess_at_1_less_all",
    "excess_low",
    "excess_high",
    "found_own",
    "events",
    "false_runs",
    "flagged_share",
    "f_own",
    "f_own_less_all",
    "f_low",
    "f_high",
)


@dataclass(frozen=True)
class Windows:
    """The task's loaded windows in the order the departures were laid: the fitting side first.

    Attributes:
        fitting: Whether the window lies on the fitting side; the rest are the validation months.
        units: Each window's unit.
        positions: Where the block holds each window.
        ends_at: When each window ends, in hours from the mission's first.
        targets: Each window's event, nought for none.
    """

    fitting: NDArray[np.bool_]
    units: NDArray[np.str_]
    positions: Ints
    ends_at: Floats
    targets: Ints

    @classmethod
    def read(cls, path: Path) -> Windows:
        """The windows a ``windows`` command wrote, refused unless indexed in their order.

        Raises:
            ValueError: If the rows are not numbered from nought in order, or name another side.
        """
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"{path}: the windows are not numbered in the order they were laid")
        sides = {row["side"] for row in rows}
        if not sides <= {"pool", "read"}:
            raise ValueError(f"{path}: a window lies on {sorted(sides - {'pool', 'read'})}")
        return cls(
            fitting=np.asarray([row["side"] == "pool" for row in rows], dtype=np.bool_),
            units=np.asarray([row["unit"] for row in rows]),
            positions=np.asarray([int(row["position"]) for row in rows], dtype=np.int64),
            ends_at=np.asarray([float(row["ends_at"]) for row in rows]),
            targets=np.asarray([round(float(row["target"])) for row in rows], dtype=np.int64),
        )

    def fitting_read(self, days: int | None = None) -> Ints:
        """The fitting windows read, those that end within the first ``days`` days if given.

        A window whose widest context reaches before the mission's first hour is not read, as
        a validation window whose context reaches a month never loaded is not: its departures
        over the longer spans would be nought for want of history, not for want of change.

        Raises:
            ValueError: If no such fitting window ends within the days given.
        """
        read = self.fitting & (self.ends_at >= WIDEST_HOURS)
        if days is not None:
            read &= self.ends_at <= days * HOURS_PER_DAY
        chosen = np.flatnonzero(read)
        if not len(chosen):
            raise ValueError(
                f"no fitting window with {WIDEST_HOURS} hours before ends by day {days}"
            )
        return chosen

    def validation(self) -> Ints:
        """Every window of the validation months, in the order laid."""
        return np.flatnonzero(~self.fitting)

    def read_of(self, indices: Ints, dropped: set[tuple[str, int]]) -> NDArray[np.bool_]:
        """Which of ``indices`` the verdict reads: every window but those ``dropped``."""
        return np.asarray(
            [(str(self.units[i]), int(self.positions[i])) not in dropped for i in indices.tolist()],
            dtype=np.bool_,
        )


@dataclass(frozen=True)
class Departures:
    """The stored context statistics of every arm, and the hour's values each arm reads.

    Attributes:
        directory: Where the departures' run keeps ``summaries.npy`` and each arm's statistics.
        cells: The one-hour published cells, ``[windows, rows, bins, features]``.
    """

    directory: Path
    cells: Floats

    def summaries(self) -> Floats:
        loaded: Floats = np.load(self.directory / "summaries.npy", mmap_mode="r")
        return loaded

    def present(self) -> NDArray[np.bool_]:
        held: NDArray[np.bool_] = ~np.isnan(np.asarray(self.summaries()[..., 0]))
        return held

    def floor(self, fit: Ints) -> Floats:
        """The least spread of a channel's history, from the fitting windows given."""
        return global_floor(np.asarray(self.summaries()[fit]))

    def of(self, arm: Arm, floor: Floats, indices: Ints) -> Floats:
        """``[len(indices), rows]``: each channel's departure under ``arm`` and ``floor``."""
        stored = [
            np.load(self.directory / f"{arm.name}-{name}.npy", mmap_mode="r")
            for name in ("centre", "spread", "loaded")
        ]
        summaries = self.summaries()
        parts = []
        for start in range(0, len(indices), CHUNK):
            chunk = indices[start : start + CHUNK]
            if arm.features == "hour":
                now = np.asarray(summaries[chunk])[:, :, None, :]
            else:
                now = bin_values(np.asarray(self.cells[chunk]))
            centre, spread, loaded = (np.asarray(array[chunk]) for array in stored)
            parts.append(
                departures_from(now, centre, spread, loaded, floor, least_context=arm.least_context)
            )
        if not parts:
            return np.zeros((0, self.summaries().shape[1]))
        return np.concatenate(parts)


def refuse_flat_months(scores: Floats, units: NDArray[np.str_]) -> None:
    """Refuse a fitting side none of whose months' scores have a spread.

    The alarm reads a score against its month's spread, a month without one against the whole
    side's, and where the side has none either, against a fixed unit: the level would then lie in
    other units than the validation months are read in. A month without spread among others is
    read as the verdict reads it.

    Raises:
        ValueError: If no fitting month's scores have a spread.
    """
    if not any(RobustScale.deviation_of(scores[units == unit]) for unit in np.unique(units)):
        raise ValueError("no fitting month scores with a spread, so the alarm has no unit")


def answered(
    departures: Departures,
    windows: Windows,
    *,
    arm: Arm,
    detector: str,
    fit: Ints,
    scored: Ints,
    present: NDArray[np.bool_],
) -> Floats:
    """The scored windows' answers: each score less the alarm set over the fitting windows.

    The floor and the alarm come from ``fit`` alone; every score is read against its month first,
    as the verdict's arms read them.
    """
    floor = departures.floor(fit)
    fit_scores = window_scores(departures.of(arm, floor, fit), present[fit], detector)
    refuse_flat_months(fit_scores, windows.units[fit])
    scored_scores = window_scores(departures.of(arm, floor, scored), present[scored], detector)
    alarm = ScoreAlarm.set_over(
        fit_scores, contamination=CONTAMINATION, units=windows.units[fit].tolist()
    )
    above: Floats = alarm.above(scored_scores, units=windows.units[scored].tolist())
    return above


def scored_side(
    departures: Departures,
    windows: Windows,
    *,
    arm: Arm,
    detector: str,
    fit: Ints,
    dropped: set[tuple[str, int]],
    present: NDArray[np.bool_],
) -> SideArrays:
    """One arm's side of the validation months, the windows the verdict does not read left out.

    The alarm reads a score against the other scores of its month given with it, so each month is
    answered whole, as the verdict's cells were, and only then are the windows ``dropped`` left out.
    """
    scored = windows.validation()
    answers = answered(
        departures, windows, arm=arm, detector=detector, fit=fit, scored=scored, present=present
    )
    read = windows.read_of(scored, dropped)
    return side_of(windows, scored[read], answers[read])


def side_of(windows: Windows, scored: Ints, answers: Floats) -> SideArrays:
    """The scored windows as one arm's side, by unit and then by place, months numbered in order."""
    order = np.lexsort((windows.positions[scored], windows.units[scored]))
    picked = scored[order]
    names = sorted(set(windows.units[picked].tolist()))
    number = {name: place for place, name in enumerate(names)}
    return SideArrays(
        units=np.asarray([number[str(u)] for u in windows.units[picked]], dtype=np.int64),
        positions=windows.positions[picked],
        events=windows.targets[picked],
        scores=answers[order],
    )


def drawn_f(tallied: Tally, weights: Floats) -> Floats:
    """The corrected event-wise F0.5 in each draw of months, ``weights`` ``[draws, months]``."""
    found = weights @ tallied.found
    events = weights @ tallied.events
    claimed = found + weights @ tallied.false_runs
    normal = weights @ tallied.normal
    quiet = np.divide(weights @ tallied.quiet, normal, out=np.zeros_like(normal), where=normal > 0)
    recall = np.divide(found, events, out=np.zeros_like(found), where=events > 0)
    precision = np.divide(found * quiet, claimed, out=np.zeros_like(found), where=claimed > 0)
    squared = BETA**2
    denominator = squared * precision + recall
    return np.divide(
        (1 + squared) * precision * recall,
        denominator,
        out=np.zeros_like(found),
        where=denominator > 0,
    )


@dataclass(frozen=True)
class Reading:
    """One arm's reading of the validation months under one fitting side.

    Attributes:
        at_one: What the alarm at 1 % of normal windows makes of each month.
        chance: The events found per month at 1 % with every month's scores turned.
        own: What the detector's own alarm makes of each month.
    """

    at_one: Tally
    chance: Floats
    own: Tally


def readings(
    sides: dict[tuple[str, str], SideArrays], rotations: int
) -> dict[tuple[str, str], Reading]:
    """Every arm's reading, the rotations shared by every arm."""
    chance = chance_by_month(sides, rotations=rotations, seed=SEED)
    return {
        key: Reading(
            at_one=tally(arrays, alarms(arrays)),
            chance=chance[key],
            own=tally(arrays, alarms(arrays, own=True)),
        )
        for key, arrays in sides.items()
    }


def curve_rows(
    days: str,
    fit_windows: int,
    read: dict[tuple[str, str], Reading],
    reference: dict[tuple[str, str], Reading],
    draws: Floats,
) -> list[dict[str, object]]:
    """The CSV rows of one fitting side, each beside the whole side over the same draws."""
    rows: list[dict[str, object]] = []
    for (detector, arm), reading in read.items():
        whole = reference[(detector, arm)]
        excess = reading.at_one.found - reading.chance
        whole_excess = whole.at_one.found - whole.chance
        excess_less = draws @ excess - draws @ whole_excess
        f_less = drawn_f(reading.own, draws) - drawn_f(whole.own, draws)
        own = reading.own
        rows.append(
            {
                "days": days,
                "detector": detector,
                "arm": arm,
                "fit_windows": fit_windows,
                "found_at_1": int(reading.at_one.found.sum()),
                "excess_at_1": f"{float(excess.sum()):.2f}",
                "excess_at_1_less_all": f"{float(excess.sum() - whole_excess.sum()):.2f}",
                "excess_low": f"{float(np.quantile(excess_less, 0.025)):.2f}",
                "excess_high": f"{float(np.quantile(excess_less, 0.975)):.2f}",
                "found_own": int(own.found.sum()),
                "events": int(own.events.sum()),
                "false_runs": int(own.false_runs.sum()),
                "flagged_share": f"{1.0 - float(own.quiet.sum()) / float(own.normal.sum()):.5f}",
                "f_own": f"{own.f_score():.4f}",
                "f_own_less_all": f"{own.f_score() - whole.own.f_score():.4f}",
                "f_low": f"{float(np.quantile(f_less, 0.025)):.4f}",
                "f_high": f"{float(np.quantile(f_less, 0.975)):.4f}",
            }
        )
    return rows


def curve(arguments: argparse.Namespace) -> None:
    """Every arm fitted on the whole fitting side and on each first N days; the CSV."""
    windows = Windows.read(arguments.windows)
    cells: Floats = np.load(arguments.cells, mmap_mode="r")
    departures = Departures(arguments.run / "departures", cells)
    present = departures.present()
    if len(present) != len(windows.units):
        raise SystemExit(
            f"{len(windows.units)} windows were listed for {len(present)} laid departures"
        )
    dropped = read_truncated(arguments.run, REFIT)
    arms = arms_of(FEATURES, SPANS)
    fits: list[tuple[str, Ints]] = [(ALL, windows.fitting_read())]
    fits += [(str(days), windows.fitting_read(days)) for days in arguments.days]
    reference: dict[tuple[str, str], Reading] = {}
    draws: Floats | None = None
    rows: list[dict[str, object]] = []
    for days, fit in fits:
        started = time.perf_counter()
        sides = {
            (detector, arm.name): scored_side(
                departures,
                windows,
                arm=arm,
                detector=detector,
                fit=fit,
                dropped=dropped,
                present=present,
            )
            for arm in arms
            for detector in LOCAL_DETECTORS
        }
        read = readings(sides, arguments.rotations)
        if days == ALL:
            reference = read
            events = next(iter(read.values())).own.events
            draws = month_draws(events, arguments.resamples, SEED)
        if draws is None:
            raise SystemExit("the whole fitting side is read first")
        rows += curve_rows(days, len(fit), read, reference, draws)
        print(
            f"{days}: {len(fit)} fitting windows, {time.perf_counter() - started:.0f} s", flush=True
        )
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def tables(out: Path) -> str:
    """The two verdicts' bars along the days, from the CSV.

    The excess's bar is the best of the twelve arms in each fitting side, read beside `mean`,
    `bins-24h`, the best on the whole side; the F-score's is the arm the verdict chose for it on
    the whole side, followed along the days.
    """
    with (out / "curve.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_days: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_days.setdefault(row["days"], []).append(row)
    lines = []
    for days, held in by_days.items():
        best = max(held, key=lambda row: float(row["excess_at_1"]))
        named = next(row for row in held if (row["detector"], row["arm"]) == ("mean", "bins-24h"))
        f_bar = next(row for row in held if (row["detector"], row["arm"]) == F_BAR)
        lines.append(
            (
                days,
                held[0]["fit_windows"],
                f"{float(best['excess_at_1']):.1f} (`{best['detector']}`, `{best['arm']}`)",
                f"{float(named['excess_at_1']):.1f}, {float(named['excess_at_1_less_all']):+.1f} "
                f"[{float(named['excess_low']):+.1f}; {float(named['excess_high']):+.1f}]",
                f"{float(f_bar['f_own']):.3f}, {float(f_bar['f_own_less_all']):+.3f} "
                f"[{float(f_bar['f_low']):+.3f}; {float(f_bar['f_high']):+.3f}]",
                f"{100 * float(f_bar['flagged_share']):.2f} %",
                f_bar["false_runs"],
                f"{f_bar['found_own']} of {f_bar['events']}",
            )
        )
    return table(
        (
            "Fitted on",
            "Windows",
            "Excess at 1 %, best of 12",
            "`mean`, `bins-24h`, less the whole side [95 %]",
            "F0.5 at own alarm, `max`, `bins-72h`, less the whole side [95 %]",
            "Normal flagged",
            "False runs",
            "Events found",
        ),
        lines,
    )


def windows_of(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """The task's loaded windows in the departures' order, fitting side first."""
    from scripts.taught_fine_tuning_pilot import bench_of

    bench = bench_of(arguments)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(WINDOW_COLUMNS)
        place = 0
        for side, labelled in (("pool", bench.pool), ("read", bench.read)):
            for each in labelled:
                window = each.window
                writer.writerow(
                    (
                        place,
                        side,
                        str(window.unit),
                        window.position,
                        repr(window.ends_at),
                        repr(float(each.target)),
                    )
                )
                place += 1
    print(f"{len(bench.pool)} fitting and {len(bench.read)} validation windows", flush=True)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("windows")
    listing.add_argument("--task", required=True)
    listing.add_argument("--task-id", required=True)
    listing.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    listing.add_argument(
        "--weights", nargs=3, action="append", default=[], metavar=("NAME", "KEY", "CHECKSUM")
    )
    listing.add_argument("--device", default="cpu")
    listing.add_argument("--out", type=Path, required=True)
    curving = commands.add_parser("curve")
    curving.add_argument("--run", type=Path, required=True)
    curving.add_argument("--cells", type=Path, required=True)
    curving.add_argument("--windows", type=Path, required=True)
    curving.add_argument("--days", type=int, nargs="+", required=True)
    curving.add_argument("--rotations", type=int, default=ROTATIONS)
    curving.add_argument("--resamples", type=int, default=RESAMPLES)
    curving.add_argument("--out", type=Path, required=True)
    tabling = commands.add_parser("tables")
    tabling.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "curve" and sorted(set(arguments.days)) != arguments.days:
        parser.error("the days are distinct and rising")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.command == "windows":
        windows_of(arguments)
    elif arguments.command == "curve":
        curve(arguments)
    else:
        print(tables(arguments.out))


if __name__ == "__main__":
    main()
