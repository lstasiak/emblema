"""Compare measures for the first mission's telemetry verdict on detectors already read.

The verdict reads the events found with 1 % of normal windows flagged. That count is half of the
measure the satellite benchmark (ESA-ADB) ranks by, chosen there with the mission's operators:
the corrected event-wise F0.5, whose precision counts the separate runs of false alarms an
operator must clear, scaled by the share of normal time left quiet. This report reads three
candidates over the no-model departure arms, on the selection months and the validation months;
the months before them are left for ``choose-f``, which chooses a bar there by the F-score:

- ``found``: the events found at 1 % of normal windows;
- ``f_own``: the corrected event-wise F0.5 at each detector's own alarm, set on the fitting side
  as every campaign sets it;
- ``f_budget``: the same at 1 % of normal windows.

For each, three properties of the measure rather than of any detector: how far chance moves it
between arms (the mean over rotations of every month's scores, the same for every arm); how well
the arms' order on one period agrees with the other's; and how many pairs of arms a paired
bootstrap over months tells apart. The CSV comes first, the tables from it.

    uv run scripts/telemetry_verdict_measure_report.py measure --dir RUN --out OUT
    uv run scripts/telemetry_verdict_measure_report.py read --out OUT
    uv run scripts/telemetry_verdict_measure_report.py power --dir RUN --out OUT
    uv run scripts/telemetry_verdict_measure_report.py choose-f --dir RUN --section early --out OUT
    uv run scripts/telemetry_verdict_measure_report.py forward --dir RUN --out OUT
        --f-bar max:bins-72h mean:bins-72h --excess-bar mean:hour-72h mean:bins-72h
    uv run scripts/telemetry_verdict_measure_report.py family --dir RUN --out OUT --verdict excess
    uv run scripts/telemetry_verdict_measure_report.py ceiling --dir RUN --taught RUN --out OUT
        --f-bar max:bins-72h mean:bins-72h --taught-arm trees:hour-6h trees:bins-24h
    uv run scripts/telemetry_verdict_measure_report.py operating-point --out OUT
        --period early RUN early --period validation RUN refit
        --task esa-mission1-events --leave-out class_22 --leave-out-in early

``best-f`` reads each detector's F-score at the best of a grid of alarms set on the months read;
``ceiling`` reads the oracle and detectors taught the events as candidates against the bars, so
a bar no detector could pass is seen before one is read against it; ``operating-point`` reads
each arm's events found above chance at 1 % beside those at its own alarm, in every period.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting import table
from scripts.taught_fine_tuning_pilot import read_answers
from scripts.telemetry_channel_local_report import (
    FEATURE_SETS,
    LOCAL,
    LOCAL_DETECTORS,
    NO_SEED,
    SEEDED,
    TAUGHT,
    TAUGHT_DETECTORS,
    Arm,
    LeftOut,
    SideArrays,
    alarm_boundary,
    arms_of,
    cell_name,
    classes_left_out,
    month_slices,
    rotation_order,
    selection_answers,
    without_events,
)
from scripts.telemetry_context_report import REFIT, SELECT, kept, read_truncated

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]

SHARE = 0.01
BETA = 0.5
MEASURES = ("found", "f_own", "f_budget")
ROTATIONS = 1000
RESAMPLES = 2000
SEED = 1
SIDES = (SELECT, REFIT)
# Two verdicts in one family under Holm: the first is read at 97.5 %, and power is planned at 0.8.
HOLM_FIRST_Z = 2.241
POWER_Z = 0.842
VERDICTS = ("f_own", "excess")


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

    def measure(self, name: str, weights: Floats | None = None) -> float:
        """``found`` or a corrected event-wise F0.5, each month counted as ``weights`` says."""
        w = np.ones(len(self.events)) if weights is None else weights
        found = float(w @ self.found)
        if name == "found":
            return found
        events = float(w @ self.events)
        recall = found / events if events else 0.0
        claimed = found + float(w @ self.false_runs)
        normal = float(w @ self.normal)
        quiet = float(w @ self.quiet) / normal if normal else 0.0
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


def alarms(arrays: SideArrays, name: str) -> Bools:
    """The windows flagged for ``name``: the detector's own alarm, or the budget of 1 %."""
    if name == "f_own":
        return arrays.scores > 0.0
    return arrays.scores > alarm_boundary(arrays.scores, arrays.events, SHARE)


def value(arrays: SideArrays, name: str) -> float:
    return tally(arrays, alarms(arrays, name)).measure(name)


def side_answers(directory: Path, section: str) -> dict[tuple[str, str], SideArrays]:
    """Every no-model arm a section holds answers for, less the windows it does not read.

    Raises:
        SystemExit: If no arm is held, or two arms hold other windows or events.
    """
    conditions = json.loads((directory / "conditions.json").read_text())
    dropped = read_truncated(directory, section)
    held = {}
    for arm in arms_of(conditions["features"], conditions["spans"]):
        for detector in LOCAL_DETECTORS:
            path = (
                directory
                / "answers"
                / f"{cell_name(section, LOCAL, detector, arm.name, NO_SEED)}.csv"
            )
            if path.exists():
                held[(detector, arm.name)] = SideArrays.of(kept(read_answers(path), dropped))
    if not held:
        raise SystemExit(f"no no-model arm holds answers on {section}")
    first = next(iter(held.values()))
    for key, arrays in held.items():
        if not (
            np.array_equal(arrays.units, first.units)
            and np.array_equal(arrays.events, first.events)
        ):
            raise SystemExit(f"{key[0]} {key[1]} holds other windows or events than the first")
    return held


def chance(
    arms: Mapping[tuple[str, str], SideArrays], *, rotations: int, seed: int
) -> dict[tuple[str, str], dict[str, float]]:
    """Each arm's measures averaged over rotations of every month's scores, one turn for all."""
    first = next(iter(arms.values()))
    slices = month_slices(first.units)
    draws = np.random.default_rng(seed)
    sums = {key: dict.fromkeys(MEASURES, 0.0) for key in arms}
    for _ in range(rotations):
        order = rotation_order(slices, draws)
        for key, arrays in arms.items():
            turned = arrays.turned(order)
            for name in MEASURES:
                sums[key][name] += value(turned, name)
    return {key: {n: s / rotations for n, s in each.items()} for key, each in sums.items()}


def chance_by_month(
    arms: Mapping[tuple[str, str], SideArrays],
    *,
    rotations: int,
    seed: int,
    alarm: str = "found",
) -> dict[tuple[str, str], Floats]:
    """Each arm's events found per month, averaged over rotations shared by every arm.

    An event counts in the month of its first window, as the tally counts it, so the excess of a
    drawn set of months is the found less the chance of the same months. ``alarm`` is ``found``
    for 1 % of normal windows or ``f_own`` for the detector's own alarm, which a rotation keeps
    flagging the same share of each month.
    """
    first = next(iter(arms.values()))
    slices = month_slices(first.units)
    draws = np.random.default_rng(seed)
    sums = {key: np.zeros(int(first.units.max()) + 1) for key in arms}
    for _ in range(rotations):
        order = rotation_order(slices, draws)
        for key, arrays in arms.items():
            turned = arrays.turned(order)
            sums[key] += tally(turned, alarms(turned, alarm)).found
    return {key: total / rotations for key, total in sums.items()}


def verdict_value(name: str, counted: Tally, by_month: Floats, weights: Floats | None) -> float:
    """``f_own``, or the excess: the events found less those found by chance, over the months."""
    if name == "f_own":
        return counted.measure("f_own", weights)
    w = np.ones(len(counted.events)) if weights is None else weights
    return float(w @ (counted.found - by_month))


def month_draws(tallies: Sequence[Tally], resamples: int, seed: int) -> Floats:
    """``[resamples, months]`` counts, months with an event and without drawn apart."""
    first = tallies[0]
    with_event = np.flatnonzero(first.events > 0)
    without = np.flatnonzero(first.events == 0)
    draws = np.random.default_rng(seed)
    counts = np.zeros((resamples, len(first.events)))
    for stratum in (with_event, without):
        if len(stratum):
            picks = draws.integers(0, len(stratum), size=(resamples, len(stratum)))
            for row in range(resamples):
                np.add.at(counts[row], stratum[picks[row]], 1.0)
    return counts


def separated(a: Tally, b: Tally, name: str, weights: Floats) -> bool:
    """Whether the 95 % interval of the paired difference over the draws excludes nought."""
    differences = np.asarray([a.measure(name, w) - b.measure(name, w) for w in weights])
    low, high = np.percentile(differences, [2.5, 97.5])
    return bool(low > 0.0 or high < 0.0)


ARM_COLUMNS = ("side", "detector", "arm", "measure", "value", "chance")
PAIR_COLUMNS = ("side", "measure", "pairs", "separated")


def measure(directory: Path, out: Path, *, rotations: int, resamples: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arm_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for side in SIDES:
        arms = side_answers(directory, side)
        expected = chance(arms, rotations=rotations, seed=SEED)
        tallies = {
            key: {name: tally(arrays, alarms(arrays, name)) for name in MEASURES}
            for key, arrays in arms.items()
        }
        for (detector, arm), each in tallies.items():
            for name in MEASURES:
                arm_rows.append(
                    {
                        "side": side,
                        "detector": detector,
                        "arm": arm,
                        "measure": name,
                        "value": f"{each[name].measure(name):.6f}",
                        "chance": f"{expected[(detector, arm)][name]:.6f}",
                    }
                )
        weights = month_draws([t["found"] for t in tallies.values()], resamples, SEED)
        for name in MEASURES:
            pairs = list(combinations(tallies, 2))
            apart = sum(
                separated(tallies[a][name], tallies[b][name], name, weights) for a, b in pairs
            )
            pair_rows.append(
                {"side": side, "measure": name, "pairs": len(pairs), "separated": apart}
            )
    write(out / "arms.csv", ARM_COLUMNS, arm_rows)
    write(out / "pairs.csv", PAIR_COLUMNS, pair_rows)


POWER_COLUMNS = ("first", "second", "verdict", "difference", "sd", "least_gain")


def power(directory: Path, out: Path, *, rotations: int, resamples: int) -> None:
    """For every pair of arms on the validation months, each verdict's paired spread over months.

    The least gain read at power 0.8 by the first test of a Holm family of two is the spread
    times the sum of the two normal quantiles, the difference taken as normal over the draws.
    """
    out.mkdir(parents=True, exist_ok=True)
    arms = side_answers(directory, REFIT)
    by_month = chance_by_month(arms, rotations=rotations, seed=SEED)
    tallies = {
        key: {name: tally(arrays, alarms(arrays, name)) for name in ("found", "f_own")}
        for key, arrays in arms.items()
    }
    weights = month_draws([t["found"] for t in tallies.values()], resamples, SEED)
    rows: list[dict[str, object]] = []
    for a, b in combinations(tallies, 2):
        for name in VERDICTS:

            def of(key: tuple[str, str], w: Floats | None, kind: str = name) -> float:
                counted = tallies[key]["f_own" if kind == "f_own" else "found"]
                return verdict_value(kind, counted, by_month[key], w)

            draws = np.asarray([of(a, w) - of(b, w) for w in weights])
            sd = float(draws.std())
            rows.append(
                {
                    "first": " ".join(a),
                    "second": " ".join(b),
                    "verdict": name,
                    "difference": f"{of(a, None) - of(b, None):.4f}",
                    "sd": f"{sd:.4f}",
                    "least_gain": f"{(HOLM_FIRST_Z + POWER_Z) * sd:.4f}",
                }
            )
    write(out / "power.csv", POWER_COLUMNS, rows)


def whole_grid(directory: Path) -> dict[tuple[str, str], SideArrays]:
    """Every no-model arm of the run's grid on the validation months.

    Raises:
        SystemExit: If any arm of the grid has no answers, so an arm cannot drop out unseen.
    """
    arms = side_answers(directory, REFIT)
    conditions = json.loads((directory / "conditions.json").read_text())
    grid = [
        (detector, arm.name)
        for arm in arms_of(conditions["features"], conditions["spans"])
        for detector in LOCAL_DETECTORS
    ]
    missing = [key for key in grid if key not in arms]
    if missing:
        raise SystemExit(f"no answers for {', '.join(' '.join(m) for m in missing)}")
    return arms


FORWARD_COLUMNS = ("verdict", "arm", "difference", "low", "high", "least_gain", "beats")
# The first test of a Holm family of two reads its interval at 97.5 %.
HOLM_FIRST_LEVEL = (1.25, 98.75)


def parse_arm(text: str) -> tuple[str, str]:
    """``<detector>:<arm>`` as the arms are keyed.

    Raises:
        argparse.ArgumentTypeError: If the text is not of that form.
    """
    detector, _, arm = text.partition(":")
    if detector not in LOCAL_DETECTORS or not arm:
        raise argparse.ArgumentTypeError(f"an arm is <detector>:<arm>, not {text!r}")
    return detector, arm


def forward(
    directory: Path,
    out: Path,
    *,
    bars: Mapping[str, Sequence[tuple[str, str]]],
    rotations: int,
    resamples: int,
    trial: int | None = None,
) -> list[dict[str, object]]:
    """Each verdict's bar, the stronger of its pair in every draw, against every other arm.

    The least gain is the median, over the pairs holding an arm of the bar, of the spread of the
    paired difference times the sum of the normal quantiles for Holm's first level and power 0.8.
    An arm beats the bar when its point gain reaches that least gain or its interval at Holm's
    first level lies above nought. With ``trial``, every month's scores are first turned by one
    offset shared by every arm.

    Raises:
        SystemExit: If a bar is not two arms, or any arm of the run's grid has no answers, so an
            arm that might beat a bar cannot drop out unseen.
    """
    for verdict, pair in bars.items():
        if len(set(pair)) != 2:
            raise SystemExit(f"the {verdict} bar is two arms, not {list(pair)}")
    out.mkdir(parents=True, exist_ok=True)
    arms = whole_grid(directory)
    if trial is not None:
        first = next(iter(arms.values()))
        order = rotation_order(month_slices(first.units), np.random.default_rng(trial))
        arms = {key: arrays.turned(order) for key, arrays in arms.items()}
    by_month = chance_by_month(arms, rotations=rotations, seed=SEED)
    tallies = {
        key: {name: tally(arrays, alarms(arrays, name)) for name in ("found", "f_own")}
        for key, arrays in arms.items()
    }
    weights = month_draws([t["found"] for t in tallies.values()], resamples, SEED)
    rows: list[dict[str, object]] = []
    for verdict, pair in bars.items():

        def of(key: tuple[str, str], w: Floats | None, kind: str = verdict) -> float:
            counted = tallies[key]["f_own" if kind == "f_own" else "found"]
            return verdict_value(kind, counted, by_month[key], w)

        drawn = {key: np.asarray([of(key, w) for w in weights]) for key in arms}
        held = {key: of(key, None) for key in arms}
        spreads = [
            float((drawn[a] - drawn[b]).std())
            for a, b in combinations(arms, 2)
            if a in pair or b in pair
        ]
        least = (HOLM_FIRST_Z + POWER_Z) * float(np.median(spreads))
        bar_drawn = np.max([drawn[key] for key in pair], axis=0)
        bar_held = max(held[key] for key in pair)
        for key in arms:
            if key in pair:
                continue
            difference = held[key] - bar_held
            low, high = np.percentile(drawn[key] - bar_drawn, HOLM_FIRST_LEVEL)
            rows.append(
                {
                    "verdict": verdict,
                    "arm": " ".join(key),
                    "difference": f"{difference:.4f}",
                    "low": f"{low:.4f}",
                    "high": f"{high:.4f}",
                    "least_gain": f"{least:.4f}",
                    "beats": int(difference >= least or low > 0.0),
                }
            )
    write(
        out / ("forward.csv" if trial is None else f"forward-trial-{trial}.csv"),
        FORWARD_COLUMNS,
        rows,
    )
    return rows


FAMILY_COLUMNS = ("verdict", "arm", "value", "best_of_rest", "sd", "least_gain")


def family(
    directory: Path, out: Path, *, verdict: str, rotations: int, resamples: int
) -> list[dict[str, object]]:
    """A bar that is the best of every no-model arm in each draw: each arm against the rest.

    Each arm stands in for a candidate against the best of the other arms in the same draw;
    the least gain is the median over the arms of the spread of that difference times the sum
    of the normal quantiles for Holm's first level and power 0.8.

    Raises:
        SystemExit: If any arm of the run's grid has no answers.
    """
    out.mkdir(parents=True, exist_ok=True)
    arms = whole_grid(directory)
    by_month = chance_by_month(arms, rotations=rotations, seed=SEED)
    source = "f_own" if verdict == "f_own" else "found"
    tallies = {key: tally(arrays, alarms(arrays, source)) for key, arrays in arms.items()}
    weights = month_draws(list(tallies.values()), resamples, SEED)
    drawn = {
        key: np.asarray([verdict_value(verdict, tallies[key], by_month[key], w) for w in weights])
        for key in arms
    }
    held = {key: verdict_value(verdict, tallies[key], by_month[key], None) for key in arms}
    spreads = {}
    rows: list[dict[str, object]] = []
    for key in arms:
        rest = [other for other in arms if other != key]
        spreads[key] = float((drawn[key] - np.max([drawn[o] for o in rest], axis=0)).std())
    least = (HOLM_FIRST_Z + POWER_Z) * float(np.median(list(spreads.values())))
    for key in arms:
        rest = [other for other in arms if other != key]
        rows.append(
            {
                "verdict": verdict,
                "arm": " ".join(key),
                "value": f"{held[key]:.4f}",
                "best_of_rest": f"{max(held[o] for o in rest):.4f}",
                "sd": f"{spreads[key]:.4f}",
                "least_gain": f"{least:.4f}",
            }
        )
    write(out / f"family-{verdict}.csv", FAMILY_COLUMNS, rows)
    return rows


# The first mission's registered verdicts: each floor's fixed part and least gain.
FLOORS = {"f_own": 0.01, "excess": 1.0}
LEAST_GAINS = {"f_own": 0.195, "excess": 10.6}
ORACLE = ("oracle", "events")
TAUGHT_SEEDS = (1, 2, 3)
CEILING_COLUMNS = (
    "verdict",
    "reference",
    "seeds",
    "value",
    "bar",
    "difference",
    "low",
    "high",
    "floor",
    "least_gain",
    "confirmed",
    "found",
    "found_own",
    "false_runs",
    "flagged",
)
# Beside each reading, what makes it: the events found at 1 % and at the detector's own alarm,
# that alarm's runs of false alarms and its share of normal windows flagged.
DESCRIBED = CEILING_COLUMNS[-4:]


def parse_taught(text: str) -> tuple[str, str]:
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
    reference: tuple[str, str],
    seeds: Sequence[int],
    like: SideArrays,
    dropped: Collection[tuple[str, int]],
) -> dict[tuple[str, str], SideArrays]:
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


def ceiling(
    validation: Path,
    taught: Path,
    out: Path,
    *,
    f_bar: Sequence[tuple[str, str]],
    references: Sequence[tuple[str, str]],
    seeds: Sequence[int] = TAUGHT_SEEDS,
    rotations: int,
    resamples: int,
    trial: int | None = None,
) -> list[dict[str, object]]:
    """The oracle and detectors taught the events, read by both verdicts against their bars.

    Each reference is read as a candidate is: its value less the bar's in every draw of months,
    the bar the stronger of the F-score's pair or the best of the twelve no-model arms; it is
    confirmed when the interval at Holm's first level lies above nought and the gain reaches the
    floor's fixed part. A seeded detector is read by its seeds' mean in every draw, and each
    seed beside. With ``trial``, every month's scores are first turned by one offset shared by
    every detector.

    Raises:
        SystemExit: If the F-score's bar is not two arms.
    """
    if len(set(f_bar)) != 2:
        raise SystemExit(f"the f_own bar is two arms, not {list(f_bar)}")
    out.mkdir(parents=True, exist_ok=True)
    arms = whole_grid(validation)
    like = next(iter(arms.values()))
    dropped = read_truncated(validation, REFIT)
    groups: dict[str, list[tuple[str, str]]] = {}
    everything: dict[tuple[str, str], SideArrays] = dict(arms)
    for reference in references:
        each = taught_answers(taught, reference, seeds, like, dropped)
        groups[" ".join(reference)] = list(each)
        everything.update(each)
    everything[ORACLE] = oracle_of(like)
    groups[ORACLE[0]] = [ORACLE]
    if trial is not None:
        order = rotation_order(month_slices(like.units), np.random.default_rng(trial))
        everything = {key: arrays.turned(order) for key, arrays in everything.items()}
    by_month = chance_by_month(everything, rotations=rotations, seed=SEED)
    tallies = {
        key: {name: tally(arrays, alarms(arrays, name)) for name in ("found", "f_own")}
        for key, arrays in everything.items()
    }
    weights = month_draws([t["found"] for t in tallies.values()], resamples, SEED)
    normal = like.events == 0
    described = {
        key: {
            "found": float(tallies[key]["found"].found.sum()),
            "found_own": float(tallies[key]["f_own"].found.sum()),
            "false_runs": float(tallies[key]["f_own"].false_runs.sum()),
            "flagged": float(alarms(arrays, "f_own")[normal].mean()),
        }
        for key, arrays in everything.items()
    }
    rows: list[dict[str, object]] = []
    for verdict in VERDICTS:

        def of(key: tuple[str, str], w: Floats | None, kind: str = verdict) -> float:
            counted = tallies[key]["f_own" if kind == "f_own" else "found"]
            return verdict_value(kind, counted, by_month[key], w)

        drawn = {key: np.asarray([of(key, w) for w in weights]) for key in everything}
        held = {key: of(key, None) for key in everything}
        bar_keys = list(f_bar) if verdict == "f_own" else list(arms)
        bar_drawn = np.max([drawn[key] for key in bar_keys], axis=0)
        bar_held = max(held[key] for key in bar_keys)
        readings = []
        for label, members in groups.items():
            readings.append((label, "all" if len(members) > 1 else "", members))
            if len(members) > 1:
                readings += [(label, key[1].split()[-1], [key]) for key in members]
        for label, seeded, members in readings:
            mean = float(np.mean([held[key] for key in members]))
            low, high = np.percentile(
                np.mean([drawn[key] for key in members], axis=0) - bar_drawn, HOLM_FIRST_LEVEL
            )
            difference = mean - bar_held
            confirmed = bool(low > 0.0 and difference >= FLOORS[verdict])

            rows.append(
                {
                    "verdict": verdict,
                    "reference": label,
                    "seeds": seeded,
                    "value": f"{mean:.4f}",
                    "bar": f"{bar_held:.4f}",
                    "difference": f"{difference:.4f}",
                    "low": f"{low:.4f}",
                    "high": f"{high:.4f}",
                    "floor": FLOORS[verdict],
                    "least_gain": LEAST_GAINS[verdict],
                    "confirmed": int(confirmed),
                    **{
                        column: f"{np.mean([described[k][column] for k in members]):.4f}"
                        for column in DESCRIBED
                    },
                }
            )
    stem = "ceiling" if trial is None else f"ceiling-trial-{trial}"
    write(out / f"{stem}.csv", CEILING_COLUMNS, rows)
    return rows


# Shares of normal windows flagged, from one in ten thousand to one in ten, evenly in the log.
SHARES = tuple(float(f"{x:.6f}") for x in np.logspace(-4, -1, 31))
BEST_F_COLUMNS = ("detector", "f_own", "best_f", "best_share", "found", "false_runs")


def best_f(
    validation: Path,
    taught: Path,
    out: Path,
    *,
    references: Sequence[tuple[str, str]],
    seeds: Sequence[int] = TAUGHT_SEEDS,
    trial: int | None = None,
) -> list[dict[str, object]]:
    """Each detector's F-score at its own alarm beside the best over a grid of alarms.

    The grid's alarm is set on the months read, so the best is what no rule set beforehand could
    do better than: a bound on what choosing the alarm can buy, the same for every detector.
    With ``trial``, every month's scores are first turned by one offset shared by every detector.
    """
    out.mkdir(parents=True, exist_ok=True)
    arms = whole_grid(validation)
    like = next(iter(arms.values()))
    dropped = read_truncated(validation, REFIT)
    everything: dict[tuple[str, str], SideArrays] = dict(arms)
    for reference in references:
        everything.update(taught_answers(taught, reference, seeds, like, dropped))
    if trial is not None:
        order = rotation_order(month_slices(like.units), np.random.default_rng(trial))
        everything = {key: arrays.turned(order) for key, arrays in everything.items()}
    rows: list[dict[str, object]] = []
    for key, arrays in everything.items():
        swept = []
        for share in SHARES:
            flags = arrays.scores > alarm_boundary(arrays.scores, arrays.events, share)
            counted = tally(arrays, flags)
            swept.append((counted.measure("f_own"), share, counted))
        best, share, counted = max(swept, key=lambda each: each[0])
        rows.append(
            {
                "detector": " ".join(key),
                "f_own": f"{value(arrays, 'f_own'):.4f}",
                "best_f": f"{best:.4f}",
                "best_share": f"{share:.5f}",
                "found": int(counted.found.sum()),
                "false_runs": int(counted.false_runs.sum()),
            }
        )
    stem = "best-f" if trial is None else f"best-f-trial-{trial}"
    write(out / f"{stem}.csv", BEST_F_COLUMNS, rows)
    return rows


def ceiling_table(rows: Sequence[Mapping[str, str]]) -> str:
    return table(
        (
            "Verdict",
            "Reference",
            "Seeds",
            "Value",
            "Bar",
            "Less the bar [97.5 %]",
            "Confirmed",
            "Found @1 %",
            "Found (own)",
            "False runs",
            "Flagged",
        ),
        [
            (
                r["verdict"],
                r["reference"],
                r["seeds"],
                r["value"],
                r["bar"],
                f"{r['difference']} [{r['low']}; {r['high']}]",
                "yes" if r["confirmed"] == "1" else "",
                r["found"],
                r["found_own"],
                r["false_runs"],
                r["flagged"],
            )
            for r in rows
        ],
    )


OPERATING_COLUMNS = (
    "period",
    "detector",
    "arm",
    "events",
    "found",
    "chance",
    "excess",
    "found_own",
    "chance_own",
    "excess_own",
    "flagged",
    "f_own",
)
AGREEMENT_MEASURES = ("excess", "excess_own", "found_own", "f_own")


def period_arms(
    directory: Path, section: str, left_out: LeftOut | None
) -> dict[tuple[str, str], SideArrays]:
    """Every no-model arm of a section, less the events left out; each arm is required."""
    answers = selection_answers(directory, section)
    return {key: SideArrays.of(without_events(rows, left_out)) for key, rows in answers.items()}


def operating_point(
    periods: Sequence[tuple[str, Mapping[tuple[str, str], SideArrays]]],
    out: Path,
    *,
    rotations: int,
    trial: int | None = None,
) -> list[dict[str, object]]:
    """Each arm's events found above chance at 1 % and at its own alarm, in every period.

    Chance at the own alarm is read by the same rotations as at 1 %: a rotation keeps the
    share each month flags, so the excess there reads where the flags fall, not how many.
    With ``trial``, every month's scores are first turned by one offset shared by every arm.
    """
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for period, held in periods:
        arms = dict(held)
        like = next(iter(arms.values()))
        if trial is not None:
            order = rotation_order(month_slices(like.units), np.random.default_rng(trial))
            arms = {key: arrays.turned(order) for key, arrays in arms.items()}
        budget = chance_by_month(arms, rotations=rotations, seed=SEED)
        own = chance_by_month(arms, rotations=rotations, seed=SEED, alarm="f_own")
        normal = like.events == 0
        for key, arrays in arms.items():
            at_budget = tally(arrays, alarms(arrays, "found"))
            own_alarm = alarms(arrays, "f_own")
            at_own = tally(arrays, own_alarm)
            found, found_own = int(at_budget.found.sum()), int(at_own.found.sum())
            rows.append(
                {
                    "period": period,
                    "detector": key[0],
                    "arm": key[1],
                    "events": len(arrays.event_ids),
                    "found": found,
                    "chance": f"{budget[key].sum():.3f}",
                    "excess": f"{found - budget[key].sum():.3f}",
                    "found_own": found_own,
                    "chance_own": f"{own[key].sum():.3f}",
                    "excess_own": f"{found_own - own[key].sum():.3f}",
                    "flagged": f"{own_alarm[normal].mean():.4f}",
                    "f_own": f"{at_own.measure('f_own'):.4f}",
                }
            )
    stem = "operating-point" if trial is None else f"operating-point-trial-{trial}"
    write(out / f"{stem}.csv", OPERATING_COLUMNS, rows)
    return rows


def agreements(rows: Sequence[Mapping[str, str]]) -> str:
    """Per measure, Spearman's agreement of the arms' order between every two periods."""
    periods = list(dict.fromkeys(r["period"] for r in rows))
    pairs = list(combinations(periods, 2))
    lines = []
    for name in AGREEMENT_MEASURES:
        by = {
            period: {
                (r["detector"], r["arm"]): float(r[name]) for r in rows if r["period"] == period
            }
            for period in periods
        }
        cells = [name]
        for a, b in pairs:
            common = sorted(set(by[a]) & set(by[b]))
            first, second = [by[a][k] for k in common], [by[b][k] for k in common]
            # An order with no spread, every arm alike, agrees with nothing.
            if len(set(first)) < 2 or len(set(second)) < 2:
                cells.append("n/a")
                continue
            cells.append(f"{spearmanr(first, second).statistic:+.2f}")
        lines.append(tuple(cells))
    return table(("Measure", *(f"{a} and {b}" for a, b in pairs)), lines)


def forward_table(rows: Sequence[Mapping[str, str]]) -> str:
    return table(
        ("Verdict", "Arm", "Less the bar", "Interval at 97.5 %", "Least gain", "Beats the bar"),
        [
            (
                r["verdict"],
                r["arm"],
                r["difference"],
                f"[{r['low']}; {r['high']}]",
                r["least_gain"],
                "yes" if r["beats"] == "1" else "",
            )
            for r in rows
        ],
    )


def power_table(rows: Sequence[Mapping[str, str]]) -> str:
    """Per verdict, the pairs' least gain at power 0.8 under Holm: least, median and largest."""
    lines = []
    for name in VERDICTS:
        gains = np.asarray([float(r["least_gain"]) for r in rows if r["verdict"] == name])
        lines.append(
            (
                name,
                str(len(gains)),
                f"{gains.min():.3f}",
                f"{np.median(gains):.3f}",
                f"{gains.max():.3f}",
            )
        )
    return table(("Verdict", "Pairs", "Least", "Median", "Largest"), lines)


CHOICE_COLUMNS = (
    "detector",
    "arm",
    "events",
    "f_own",
    "found_own",
    "chance",
    "excess",
    "false_runs",
    "flagged",
    "chosen",
)


@dataclass(frozen=True)
class FReading:
    """An arm on a choosing section, by the benchmark's F-score and what it is made of.

    Attributes:
        detector: How the channels' departures become the window's score.
        arm: The feature set and span.
        events: The events the side scores.
        f_own: The corrected event-wise F0.5 at the arm's own alarm.
        found: The events found at that alarm.
        chance: The events found at 1 % on average with every month's scores turned.
        excess: The events found at 1 % less ``chance``.
        false_runs: The runs of false alarms at the arm's own alarm.
        flagged: The share of normal windows the arm's own alarm flags.
    """

    detector: str
    arm: str
    events: int
    f_own: float
    found: int
    chance: float
    excess: float
    false_runs: int
    flagged: float


def f_reading(key: tuple[str, str], arrays: SideArrays, expected: float) -> FReading:
    own = alarms(arrays, "f_own")
    counted = tally(arrays, own)
    at_budget = int(tally(arrays, alarms(arrays, "found")).found.sum())
    normal = arrays.events == 0
    return FReading(
        detector=key[0],
        arm=key[1],
        events=len(arrays.event_ids),
        f_own=counted.measure("f_own"),
        found=int(counted.found.sum()),
        chance=expected,
        excess=at_budget - expected,
        false_runs=int(counted.false_runs.sum()),
        flagged=float(own[normal].mean()),
    )


def choose_by_f(readings: Sequence[FReading]) -> FReading:
    """The highest F-score; then the most found; then the shorter span, the hour, the largest.

    Raises:
        ValueError: If there is no reading.
    """
    if not readings:
        raise ValueError("no arm to choose from")

    def order(reading: FReading) -> tuple[float, int, int, int, int]:
        arm = Arm.parse(reading.arm)
        return (
            round(reading.f_own, 9),
            reading.found,
            -arm.hours,
            -FEATURE_SETS.index(arm.features),
            -LOCAL_DETECTORS.index(reading.detector),
        )

    return max(readings, key=order)


def choose_f(
    directory: Path,
    out: Path,
    *,
    section: str,
    left_out: LeftOut | None = None,
    rotations: int = ROTATIONS,
    trial: int | None = None,
) -> str:
    """Write the choice by the benchmark's F-score to ``out``: the CSV, its table, the arm chosen.

    With ``trial``, every month's scores are first turned by one offset shared by every arm.
    """
    out.mkdir(parents=True, exist_ok=True)
    answers = selection_answers(directory, section)
    arms = {key: SideArrays.of(without_events(rows, left_out)) for key, rows in answers.items()}
    if trial is not None:
        first = next(iter(arms.values()))
        order = rotation_order(month_slices(first.units), np.random.default_rng(trial))
        arms = {key: arrays.turned(order) for key, arrays in arms.items()}
    by_month = chance_by_month(arms, rotations=rotations, seed=SEED)
    readings = [f_reading(key, arrays, float(by_month[key].sum())) for key, arrays in arms.items()]
    chosen = choose_by_f(readings)
    stem = f"f-{section}{left_out.stem if left_out else ''}"
    if trial is not None:
        stem += f"-trial-{trial}"
    rows = [
        {
            "detector": r.detector,
            "arm": r.arm,
            "events": r.events,
            "f_own": f"{r.f_own:.4f}",
            "found_own": r.found,
            "chance": f"{r.chance:.3f}",
            "excess": f"{r.excess:.3f}",
            "false_runs": r.false_runs,
            "flagged": f"{r.flagged:.4f}",
            "chosen": int(r == chosen),
        }
        for r in readings
    ]
    write(out / f"{stem}.csv", CHOICE_COLUMNS, rows)
    text = table(
        (
            "Detector",
            "Arm",
            "F own",
            "Found (own)",
            "Excess @1 %",
            "False runs",
            "Flagged",
            "Chosen",
        ),
        [
            (
                r["detector"],
                r["arm"],
                r["f_own"],
                f"{r['found_own']} of {r['events']}",
                r["excess"],
                r["false_runs"],
                r["flagged"],
                "yes" if r["chosen"] == "1" else "",
            )
            for r in rows_of(out / f"{stem}.csv")
        ],
    )
    (out / f"{stem}.md").write_text(text + "\n")
    record = {**chosen.__dict__, "section": section, "rotations": rotations, "trial": trial}
    record["left_out"] = sorted(left_out.classes) if left_out else []
    (out / f"{stem}-chosen.json").write_text(json.dumps(record, indent=2) + "\n")
    return text


def write(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def properties(
    arm_rows: Sequence[Mapping[str, str]], pair_rows: Sequence[Mapping[str, str]]
) -> list[tuple[str, ...]]:
    """Per side and measure, the properties the choice of a measure is judged by.

    Chance's spread over the arms against the arms' own, the correlation of value and chance,
    the agreement of the arms' order between the two sides, and the pairs told apart.
    """
    out: list[tuple[str, ...]] = []

    def by(side: str, name: str) -> dict[tuple[str, str], tuple[float, float]]:
        return {
            (r["detector"], r["arm"]): (float(r["value"]), float(r["chance"]))
            for r in arm_rows
            if r["side"] == side and r["measure"] == name
        }

    for name in MEASURES:
        sides = {side: by(side, name) for side in SIDES}
        common = sorted(set(sides[SELECT]) & set(sides[REFIT]))
        agreement = spearmanr(
            [sides[SELECT][k][0] for k in common], [sides[REFIT][k][0] for k in common]
        ).statistic
        for side in SIDES:
            values = np.asarray([v for v, _ in sides[side].values()])
            chances = np.asarray([c for _, c in sides[side].values()])
            spread = values.max() - values.min()
            pairs = next(r for r in pair_rows if r["side"] == side and r["measure"] == name)
            out.append(
                (
                    name,
                    side,
                    str(len(values)),
                    f"{chances.max() - chances.min():.3f} / {spread:.3f}",
                    f"{np.corrcoef(values, chances)[0, 1]:+.2f}",
                    f"{agreement:+.2f} ({len(common)})",
                    f"{pairs['separated']} of {pairs['pairs']}",
                )
            )
    return out


def read(out: Path) -> str:
    text = table(
        (
            "Measure",
            "Side",
            "Arms",
            "Chance spread / arms' spread",
            "Value vs chance",
            "Order agreement",
            "Pairs apart",
        ),
        properties(rows_of(out / "arms.csv"), rows_of(out / "pairs.csv")),
    )
    (out / "tables.md").write_text(text + "\n")
    return text


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--dir", type=Path, required=True)
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--rotations", type=int, default=ROTATIONS)
    measuring.add_argument("--resamples", type=int, default=RESAMPLES)
    reading = commands.add_parser("read")
    reading.add_argument("--out", type=Path, required=True)
    by_f = commands.add_parser("choose-f")
    by_f.add_argument("--dir", type=Path, required=True)
    by_f.add_argument("--section", required=True)
    by_f.add_argument("--out", type=Path, required=True)
    by_f.add_argument("--task", help="the task whose event classes --leave-out names")
    by_f.add_argument("--leave-out", nargs="+", default=[], metavar="CLASS")
    by_f.add_argument("--rotations", type=int, default=ROTATIONS)
    by_f.add_argument("--trial", type=int, metavar="SEED", help="turn every month's scores first")
    forwarding = commands.add_parser("forward")
    forwarding.add_argument("--dir", type=Path, required=True)
    forwarding.add_argument("--out", type=Path, required=True)
    forwarding.add_argument("--f-bar", type=parse_arm, nargs=2, required=True)
    forwarding.add_argument("--excess-bar", type=parse_arm, nargs=2, required=True)
    forwarding.add_argument("--rotations", type=int, default=ROTATIONS)
    forwarding.add_argument("--resamples", type=int, default=RESAMPLES)
    forwarding.add_argument(
        "--trial", type=int, metavar="SEED", help="turn every month's scores first"
    )
    familying = commands.add_parser("family")
    familying.add_argument("--dir", type=Path, required=True)
    familying.add_argument("--out", type=Path, required=True)
    familying.add_argument("--verdict", choices=VERDICTS, required=True)
    familying.add_argument("--rotations", type=int, default=ROTATIONS)
    familying.add_argument("--resamples", type=int, default=RESAMPLES)
    ceiling_parser = commands.add_parser("ceiling")
    ceiling_parser.add_argument("--dir", type=Path, required=True)
    ceiling_parser.add_argument("--taught", type=Path, required=True)
    ceiling_parser.add_argument("--out", type=Path, required=True)
    ceiling_parser.add_argument("--f-bar", type=parse_arm, nargs=2, required=True)
    ceiling_parser.add_argument("--taught-arm", type=parse_taught, nargs="+", required=True)
    ceiling_parser.add_argument("--seeds", type=int, nargs="+", default=list(TAUGHT_SEEDS))
    ceiling_parser.add_argument("--rotations", type=int, default=ROTATIONS)
    ceiling_parser.add_argument("--resamples", type=int, default=RESAMPLES)
    ceiling_parser.add_argument(
        "--trial", type=int, metavar="SEED", help="turn every month's scores first"
    )
    best = commands.add_parser("best-f")
    best.add_argument("--dir", type=Path, required=True)
    best.add_argument("--taught", type=Path, required=True)
    best.add_argument("--out", type=Path, required=True)
    best.add_argument("--taught-arm", type=parse_taught, nargs="+", required=True)
    best.add_argument("--seeds", type=int, nargs="+", default=list(TAUGHT_SEEDS))
    best.add_argument("--trial", type=int, metavar="SEED", help="turn every month's scores first")
    operating = commands.add_parser("operating-point")
    operating.add_argument(
        "--period", nargs=3, action="append", required=True, metavar=("NAME", "DIR", "SECTION")
    )
    operating.add_argument("--out", type=Path, required=True)
    operating.add_argument("--task", help="the task whose event classes --leave-out names")
    operating.add_argument("--leave-out", nargs="+", default=[], metavar="CLASS")
    operating.add_argument("--leave-out-in", nargs="+", default=[], metavar="NAME")
    operating.add_argument("--rotations", type=int, default=ROTATIONS)
    operating.add_argument(
        "--trial", type=int, metavar="SEED", help="turn every month's scores first"
    )
    powering = commands.add_parser("power")
    powering.add_argument("--dir", type=Path, required=True)
    powering.add_argument("--out", type=Path, required=True)
    powering.add_argument("--rotations", type=int, default=ROTATIONS)
    powering.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args(argv)
    if arguments.command == "choose-f":
        print(
            choose_f(
                arguments.dir,
                arguments.out,
                section=arguments.section,
                left_out=classes_left_out(arguments.task, arguments.leave_out),
                rotations=arguments.rotations,
                trial=arguments.trial,
            )
        )
        return
    if arguments.command == "forward":
        forward(
            arguments.dir,
            arguments.out,
            bars={"f_own": arguments.f_bar, "excess": arguments.excess_bar},
            rotations=arguments.rotations,
            resamples=arguments.resamples,
            trial=arguments.trial,
        )
        stem = "forward" if arguments.trial is None else f"forward-trial-{arguments.trial}"
        text = forward_table(rows_of(arguments.out / f"{stem}.csv"))
        (arguments.out / f"{stem}.md").write_text(text + "\n")
        print(text)
        return
    if arguments.command == "ceiling":
        ceiling(
            arguments.dir,
            arguments.taught,
            arguments.out,
            f_bar=arguments.f_bar,
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
        return
    if arguments.command == "best-f":
        best_f(
            arguments.dir,
            arguments.taught,
            arguments.out,
            references=arguments.taught_arm,
            seeds=arguments.seeds,
            trial=arguments.trial,
        )
        stem = "best-f" if arguments.trial is None else f"best-f-trial-{arguments.trial}"
        text = table(
            ("Detector", "F own", "Best F", "At share", "Found", "False runs"),
            [tuple(r.values()) for r in rows_of(arguments.out / f"{stem}.csv")],
        )
        (arguments.out / f"{stem}.md").write_text(text + "\n")
        print(text)
        return
    if arguments.command == "operating-point":
        names = [name for name, _, _ in arguments.period]
        if bool(arguments.leave_out) != bool(arguments.leave_out_in):
            raise SystemExit("--leave-out names classes and --leave-out-in periods: give both")
        unknown = sorted(set(arguments.leave_out_in) - set(names))
        if unknown:
            raise SystemExit(f"--leave-out-in names no period: {', '.join(unknown)}")
        left_out = classes_left_out(arguments.task, arguments.leave_out)
        periods = [
            (
                name,
                period_arms(
                    Path(directory),
                    section,
                    left_out if name in arguments.leave_out_in else None,
                ),
            )
            for name, directory, section in arguments.period
        ]
        operating_point(
            periods, arguments.out, rotations=arguments.rotations, trial=arguments.trial
        )
        stem = (
            "operating-point"
            if arguments.trial is None
            else f"operating-point-trial-{arguments.trial}"
        )
        text = agreements(rows_of(arguments.out / f"{stem}.csv"))
        (arguments.out / f"{stem}.md").write_text(text + "\n")
        print(text)
        return
    if arguments.command == "family":
        rows = family(
            arguments.dir,
            arguments.out,
            verdict=arguments.verdict,
            rotations=arguments.rotations,
            resamples=arguments.resamples,
        )
        text = table(
            ("Arm", "Value", "Best of the rest", "SD", "Least gain"),
            [
                (
                    str(r["arm"]),
                    str(r["value"]),
                    str(r["best_of_rest"]),
                    str(r["sd"]),
                    str(r["least_gain"]),
                )
                for r in rows
            ],
        )
        (arguments.out / f"family-{arguments.verdict}.md").write_text(text + "\n")
        print(text)
        return
    if arguments.command == "power":
        power(
            arguments.dir,
            arguments.out,
            rotations=arguments.rotations,
            resamples=arguments.resamples,
        )
        text = power_table(rows_of(arguments.out / "power.csv"))
        (arguments.out / "power.md").write_text(text + "\n")
        print(text)
        return
    if arguments.command == "measure":
        measure(
            arguments.dir,
            arguments.out,
            rotations=arguments.rotations,
            resamples=arguments.resamples,
        )
    print(read(arguments.out))


if __name__ == "__main__":  # pragma: no cover - entrypoint
    main()
