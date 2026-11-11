"""Read the first mission's telemetry verdict on detectors already read.

The verdict is the excess: the events found with 1 % of normal windows flagged, less those found
when every month's scores are turned, against the best of the twelve no-model departure arms in
each draw of months. Beside every reading: the events found and the excess at the detector's own
alarm, the satellite benchmark's corrected event-wise F0.5 there (ESA-ADB), its runs of false
alarms and its share of normal windows flagged. The CSV comes first, the tables from it.

    uv run scripts/telemetry_verdict_measure_report.py family --dir RUN --out OUT
    uv run scripts/telemetry_verdict_measure_report.py ceiling --dir RUN --taught RUN --out OUT
        --taught-arm trees:hour-6h trees:bins-24h

``family`` reads each departure arm against the best of the others, which gives the least gain;
``ceiling`` reads the oracle and detectors taught the events as candidates against the bar.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting import table
from scripts.taught_fine_tuning_pilot import read_answers
from scripts.telemetry_channel_local_report import (
    NO_SEED,
    SEEDED,
    TAUGHT,
    TAUGHT_DETECTORS,
    SideArrays,
    alarm_boundary,
    cell_name,
    month_slices,
    rotation_order,
    selection_answers,
)
from scripts.telemetry_context_report import REFIT, kept, read_truncated

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]
Key = tuple[str, str]

SHARE = 0.01
BETA = 0.5
ROTATIONS = 1000
RESAMPLES = 2000
SEED = 1
# The level and least gain were set for a family of two verdicts under Holm and are kept for the
# excess read alone, so dropping the other verdict loosens nothing: the first test of the family
# reads its interval at 97.5 %, and power is planned at 0.8.
HOLM_FIRST_Z = 2.241
POWER_Z = 0.842
HOLM_FIRST_LEVEL = (1.25, 98.75)
FLOOR = 1.0
LEAST_GAIN = 10.6
ORACLE = ("oracle", "events")
TAUGHT_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class Tally:
    """What an alarm makes of a side, per month, so months can be drawn again.

    Attributes:
        events: The events whose first window lies in each month.
        found: Of those, the events with a flagged window anywhere.
        false_runs: The runs of consecutive flagged windows touching no event, per month.
        normal: The normal windows per month.
        quiet: The normal windows left unflagged, per month.
    """

    events: Ints
    found: Ints
    false_runs: Ints
    normal: Ints
    quiet: Ints

    def f_score(self) -> float:
        """The corrected event-wise F0.5: precision counts runs of false alarms and quiet time."""
        found = float(self.found.sum())
        events = float(self.events.sum())
        recall = found / events if events else 0.0
        claimed = found + float(self.false_runs.sum())
        normal = float(self.normal.sum())
        quiet = float(self.quiet.sum()) / normal if normal else 0.0
        precision = found / claimed * quiet if claimed else 0.0
        if precision + recall == 0.0:
            return 0.0
        return (1 + BETA**2) * precision * recall / (BETA**2 * precision + recall)


def tally(arrays: SideArrays, flagged: Bools) -> Tally:
    """The side's tally under ``flagged``, months numbered as ``arrays.units`` numbers them."""
    months = int(arrays.units.max()) + 1
    ids = arrays.event_ids
    first = np.asarray([arrays.units[np.argmax(arrays.events == e)] for e in ids], dtype=np.int64)
    hit = np.isin(ids, np.unique(arrays.events[flagged & (arrays.events > 0)]))
    same_run = np.concatenate(
        (
            [False],
            (arrays.units[1:] == arrays.units[:-1])
            & (arrays.positions[1:] == arrays.positions[:-1] + 1)
            & flagged[:-1],
        )
    )
    starts = flagged & ~same_run
    run_of = np.cumsum(starts) - 1
    run_holds_event = np.zeros(int(starts.sum()), dtype=bool)
    np.logical_or.at(run_holds_event, run_of[flagged], arrays.events[flagged] > 0)
    run_month = arrays.units[starts]
    normal = arrays.events == 0
    return Tally(
        events=np.bincount(first, minlength=months),
        found=np.bincount(first[hit], minlength=months),
        false_runs=np.bincount(run_month[~run_holds_event], minlength=months),
        normal=np.bincount(arrays.units[normal], minlength=months),
        quiet=np.bincount(arrays.units[normal & ~flagged], minlength=months),
    )


def alarms(arrays: SideArrays, *, own: bool = False) -> Bools:
    """The windows flagged at 1 % of normal windows, or at the detector's own alarm."""
    if own:
        return arrays.scores > 0.0
    return arrays.scores > alarm_boundary(arrays.scores, arrays.events, SHARE)


def turned(arms: Mapping[Key, SideArrays], trial: int | None) -> dict[Key, SideArrays]:
    """The detectors as read, or with every month's scores turned by one offset for all."""
    if trial is None:
        return dict(arms)
    like = next(iter(arms.values()))
    order = rotation_order(month_slices(like.units), np.random.default_rng(trial))
    return {key: arrays.turned(order) for key, arrays in arms.items()}


def chance_by_month(
    arms: Mapping[Key, SideArrays], *, rotations: int, seed: int, own: bool = False
) -> dict[Key, Floats]:
    """Each detector's events found per month, averaged over rotations shared by every one.

    An event counts in the month of its first window, as the tally counts it, so the excess of a
    drawn set of months is the found less the chance of the same months. At the own alarm a
    rotation keeps the share each month flags, so the excess there reads where the flags fall.
    """
    first = next(iter(arms.values()))
    slices = month_slices(first.units)
    draws = np.random.default_rng(seed)
    sums = {key: np.zeros(int(first.units.max()) + 1) for key in arms}
    for _ in range(rotations):
        order = rotation_order(slices, draws)
        for key, arrays in arms.items():
            again = arrays.turned(order)
            sums[key] += tally(again, alarms(again, own=own)).found
    return {key: total / rotations for key, total in sums.items()}


def excess(found: Ints, chance: Floats, weights: Floats | None = None) -> float:
    """The events found less those found by chance, each month counted as ``weights`` says."""
    w = np.ones(len(found)) if weights is None else weights
    return float(w @ (found - chance))


def month_draws(events: Ints, resamples: int, seed: int) -> Floats:
    """``[resamples, months]`` counts, months with an event and without drawn apart."""
    with_event = np.flatnonzero(events > 0)
    without = np.flatnonzero(events == 0)
    draws = np.random.default_rng(seed)
    counts = np.zeros((resamples, len(events)))
    for stratum in (with_event, without):
        if len(stratum):
            picks = draws.integers(0, len(stratum), size=(resamples, len(stratum)))
            for row in range(resamples):
                np.add.at(counts[row], stratum[picks[row]], 1.0)
    return counts


def departure_arms(directory: Path) -> dict[Key, SideArrays]:
    """Every no-model departure arm on the validation months; each one is required."""
    return {key: SideArrays.of(rows) for key, rows in selection_answers(directory, REFIT).items()}


FAMILY_COLUMNS = ("arm", "value", "best_of_rest", "sd", "least_gain")


def family(
    directory: Path, out: Path, *, rotations: int, resamples: int
) -> list[dict[str, object]]:
    """A bar that is the best of every no-model arm in each draw: each arm against the rest.

    Each arm stands in for a candidate against the best of the other arms in the same draw; the
    least gain is the median over the arms of the spread of that difference times the sum of the
    normal quantiles for the reading's level and power 0.8.
    """
    out.mkdir(parents=True, exist_ok=True)
    arms = departure_arms(directory)
    by_month = chance_by_month(arms, rotations=rotations, seed=SEED)
    tallies = {key: tally(arrays, alarms(arrays)) for key, arrays in arms.items()}
    found = {key: counted.found for key, counted in tallies.items()}
    weights = month_draws(next(iter(tallies.values())).events, resamples, SEED)
    drawn = {
        key: np.asarray([excess(found[key], by_month[key], w) for w in weights]) for key in arms
    }
    held = {key: excess(found[key], by_month[key]) for key in arms}
    spreads = {}
    for key in arms:
        rest = [other for other in arms if other != key]
        spreads[key] = float((drawn[key] - np.max([drawn[o] for o in rest], axis=0)).std())
    least = (HOLM_FIRST_Z + POWER_Z) * float(np.median(list(spreads.values())))
    rows: list[dict[str, object]] = []
    for key in arms:
        rest = [other for other in arms if other != key]
        rows.append(
            {
                "arm": " ".join(key),
                "value": f"{held[key]:.4f}",
                "best_of_rest": f"{max(held[o] for o in rest):.4f}",
                "sd": f"{spreads[key]:.4f}",
                "least_gain": f"{least:.4f}",
            }
        )
    write(out / "family.csv", FAMILY_COLUMNS, rows)
    return rows


def parse_taught(text: str) -> Key:
    """``<taught detector>:<arm>``, as the taught cells are named.

    Raises:
        argparse.ArgumentTypeError: If the text is not of that form.
    """
    detector, _, arm = text.partition(":")
    if detector not in TAUGHT_DETECTORS or not arm:
        raise argparse.ArgumentTypeError(f"a taught arm is <detector>:<arm>, not {text!r}")
    return detector, arm


def taught_answers(
    directory: Path,
    reference: Key,
    seeds: Sequence[int],
    like: SideArrays,
    dropped: Collection[tuple[str, int]],
) -> dict[Key, SideArrays]:
    """A taught detector's answers on the validation months, one per seed it draws.

    Raises:
        SystemExit: If a seed's cell is missing, or it scores other windows or events than
            ``like``, so a reference is never read on another set of windows than its bar.
    """
    detector, arm = reference
    held = {}
    for seed in seeds if detector in SEEDED else (NO_SEED,):
        name = cell_name(REFIT, TAUGHT, detector, arm, seed)
        path = directory / "answers" / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"{name} is missing")
        arrays = SideArrays.of(kept(read_answers(path), dropped))
        if not (
            np.array_equal(arrays.units, like.units)
            and np.array_equal(arrays.positions, like.positions)
            and np.array_equal(arrays.events, like.events)
        ):
            raise SystemExit(f"{name} scores other windows or events than the no-model arms")
        held[(detector, arm if seed == NO_SEED else f"{arm} s{seed}")] = arrays
    return held


def oracle_of(like: SideArrays) -> SideArrays:
    """Every event window above the alarm and every normal window below it."""
    return replace(like, scores=np.where(like.events > 0, 1.0, -1.0))


CEILING_COLUMNS = (
    "reference",
    "seeds",
    "value",
    "bar",
    "difference",
    "low",
    "high",
    "confirmed",
    "found",
    "found_own",
    "excess_own",
    "f_own",
    "false_runs",
    "flagged",
)
# Beside each reading, what makes it and what an operator would meet at the detector's own alarm.
DESCRIBED = CEILING_COLUMNS[-6:]


def ceiling(
    validation: Path,
    taught: Path,
    out: Path,
    *,
    references: Sequence[Key],
    seeds: Sequence[int] = TAUGHT_SEEDS,
    rotations: int,
    resamples: int,
    trial: int | None = None,
) -> list[dict[str, object]]:
    """The oracle and detectors taught the events, read as candidates against the bar.

    A reference's excess less the best departure arm's, in every draw of months; confirmed when
    the interval at 97.5 % lies above nought and the gain reaches the floor's fixed part. A seeded
    detector is read by its seeds' mean in every draw, and each seed beside. With ``trial``,
    every month's scores are first turned by one offset shared by every detector.
    """
    out.mkdir(parents=True, exist_ok=True)
    arms = departure_arms(validation)
    like = next(iter(arms.values()))
    dropped = read_truncated(validation, REFIT)
    groups: dict[str, list[Key]] = {}
    everything: dict[Key, SideArrays] = dict(arms)
    for reference in references:
        each = taught_answers(taught, reference, seeds, like, dropped)
        groups[" ".join(reference)] = list(each)
        everything.update(each)
    everything[ORACLE] = oracle_of(like)
    groups[ORACLE[0]] = [ORACLE]
    everything = turned(everything, trial)
    by_month = chance_by_month(everything, rotations=rotations, seed=SEED)
    by_month_own = chance_by_month(everything, rotations=rotations, seed=SEED, own=True)
    at_budget = {key: tally(arrays, alarms(arrays)) for key, arrays in everything.items()}
    at_own = {key: tally(arrays, alarms(arrays, own=True)) for key, arrays in everything.items()}
    weights = month_draws(next(iter(at_budget.values())).events, resamples, SEED)
    normal = like.events == 0
    described = {
        key: {
            "found": float(at_budget[key].found.sum()),
            "found_own": float(at_own[key].found.sum()),
            "excess_own": excess(at_own[key].found, by_month_own[key]),
            "f_own": at_own[key].f_score(),
            "false_runs": float(at_own[key].false_runs.sum()),
            "flagged": float(alarms(arrays, own=True)[normal].mean()),
        }
        for key, arrays in everything.items()
    }
    drawn = {
        key: np.asarray([excess(at_budget[key].found, by_month[key], w) for w in weights])
        for key in everything
    }
    held = {key: excess(at_budget[key].found, by_month[key]) for key in everything}
    bar_drawn = np.max([drawn[key] for key in arms], axis=0)
    bar_held = max(held[key] for key in arms)
    readings = []
    for label, members in groups.items():
        readings.append((label, "all" if len(members) > 1 else "", members))
        if len(members) > 1:
            readings += [(label, key[1].split()[-1], [key]) for key in members]
    rows: list[dict[str, object]] = []
    for label, seeded, members in readings:
        mean = float(np.mean([held[key] for key in members]))
        low, high = np.percentile(
            np.mean([drawn[key] for key in members], axis=0) - bar_drawn, HOLM_FIRST_LEVEL
        )
        difference = mean - bar_held
        rows.append(
            {
                "reference": label,
                "seeds": seeded,
                "value": f"{mean:.4f}",
                "bar": f"{bar_held:.4f}",
                "difference": f"{difference:.4f}",
                "low": f"{low:.4f}",
                "high": f"{high:.4f}",
                "confirmed": int(bool(low > 0.0 and difference >= FLOOR)),
                **{
                    column: f"{np.mean([described[k][column] for k in members]):.4f}"
                    for column in DESCRIBED
                },
            }
        )
    stem = "ceiling" if trial is None else f"ceiling-trial-{trial}"
    write(out / f"{stem}.csv", CEILING_COLUMNS, rows)
    return rows


def ceiling_table(rows: Sequence[Mapping[str, str]]) -> str:
    return table(
        (
            "Reference",
            "Seeds",
            "Excess",
            "Bar",
            "Less the bar [97.5 %]",
            "Confirmed",
            "Found @1 %",
            "Found (own)",
            "Excess (own)",
            "F own",
            "False runs",
            "Flagged",
        ),
        [
            (
                r["reference"],
                r["seeds"],
                r["value"],
                r["bar"],
                f"{r['difference']} [{r['low']}; {r['high']}]",
                "yes" if r["confirmed"] == "1" else "",
                r["found"],
                r["found_own"],
                r["excess_own"],
                r["f_own"],
                r["false_runs"],
                r["flagged"],
            )
            for r in rows
        ],
    )


def write(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dir", type=Path, required=True)
    shared.add_argument("--out", type=Path, required=True)
    shared.add_argument("--rotations", type=int, default=ROTATIONS)
    shared.add_argument("--resamples", type=int, default=RESAMPLES)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("family", parents=[shared])
    reading = commands.add_parser("ceiling", parents=[shared])
    reading.add_argument("--taught", type=Path, required=True)
    reading.add_argument("--taught-arm", type=parse_taught, nargs="+", required=True)
    reading.add_argument("--seeds", type=int, nargs="+", default=list(TAUGHT_SEEDS))
    reading.add_argument(
        "--trial", type=int, metavar="SEED", help="turn every month's scores first"
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "family":
        family(
            arguments.dir,
            arguments.out,
            rotations=arguments.rotations,
            resamples=arguments.resamples,
        )
        text = table(
            ("Arm", "Value", "Best of the rest", "SD", "Least gain"),
            [tuple(r.values()) for r in rows_of(arguments.out / "family.csv")],
        )
        (arguments.out / "family.md").write_text(text + "\n")
        print(text)
        return
    ceiling(
        arguments.dir,
        arguments.taught,
        arguments.out,
        references=arguments.taught_arm,
        seeds=arguments.seeds,
        rotations=arguments.rotations,
        resamples=arguments.resamples,
        trial=arguments.trial,
    )
    stem = "ceiling" if arguments.trial is None else f"ceiling-trial-{arguments.trial}"
    text = ceiling_table(rows_of(arguments.out / f"{stem}.csv"))
    (arguments.out / f"{stem}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
