"""Which validation events of the first mission anything finds, and what a finer cell could add.

The detectors' answers already on disk are read per event at 1 % of normal windows. An event is
filed under the first family that finds it: the channels' departures from their recent hours, any
other detector with no label, a detector taught labels, or none. Summing detectors overstates what
is reachable, since each adds its own false alarms and long events catch chance alarms, so the
events the departures miss are also counted under rotations of the scores within months, one
offset per month shared by every detector, which keeps detectors that agree in agreement.

    uv run scripts/telemetry_ceiling_report.py preflight --sources FILE --out DIR
    uv run scripts/telemetry_ceiling_report.py measure --side refit --sources FILE --out DIR
        [--trial]
    uv run scripts/telemetry_ceiling_report.py resolution --sources FILE --out DIR [--trial]
    uv run scripts/telemetry_ceiling_report.py post-hoc --sources FILE --out DIR
    uv run scripts/telemetry_ceiling_report.py read --dir DIR

``post-hoc`` holds what was asked after the declared reading and decides nothing: the departures
under the rotations, the fusion against better departures, and each member's share in it.

``measure`` writes CSV only; ``read`` makes the tables from it. ``--trial`` rotates the targets
within months first, so the whole path runs with no reading of an event, and prints counts.
"""

import argparse
import csv
import math
import re
import sys
import tomllib
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.detection_power_report import (
    Structure,
    confirms_recall,
    event_recalls,
    recall_interval,
)
from scripts.reporting import table
from scripts.telemetry_channel_local_report import (
    MAX,
    UNFOUND_BEFORE,
    event_annotations,
    window_scores,
)
from scripts.telemetry_context_report import read_truncated

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
Bools = NDArray[np.bool_]
Window = tuple[str, int]

SHARE = 0.01
MISSION = "ESA-Mission1"
VALIDATION_STARTS = "2007-01"
SELECTION_STARTS = "2004-01"
# The channels the first mission's corpus reads, in the order of the departures' rows.
CHANNEL_ROWS = (*range(12, 53), *range(70, 77))
# An answer file's rows start at this index of the arrays over every loaded window.
FIRST_REFIT_INDEX = 61_324
LEAST_GAIN, FLOOR = 0.18, 0.016
QUEUE_EVENTS, QUEUE_CHANNEL_SETS, CHANCE_RATE = 4, 2, 0.05
FUSION_RESAMPLES = 10_000

DEPARTURES, OTHER, TAUGHT, NOTHING = "departures", "other", "taught", "none"
REFIT, SELECT = "refit", "select"
SEEN, UNSEEN = "seen", "unseen"
SEEDED = re.compile(r"-s(?P<seed>\d+)(?P<selection>@selection)?$")
FREE_CANDIDATES = (
    "cells_knn",
    "counts_knn",
    "frozen_knn",
    "interpolation_error",
    "linear_answer_error",
    "pretext_error",
    "summaries_knn",
    "untrained_knn",
)
CHANNEL_LOCAL = "channel_local@reference=unit"
# One representative per family of detector with no label; the variants would swamp the fusion.
REPRESENTATIVES = (
    ("f9c2680d", CHANNEL_LOCAL),
    ("f9c2680d", "cells_knn@distance=farthest,reference=unit"),
    ("fdd31528", "pretext_error@error=largest,hidden=tail,reference=unit"),
    ("fdd31528", "interpolation_error@error=largest,hidden=tail,reference=unit"),
    ("3bed588b", "frozen_knn@distance=whitened,reference=unit"),
)
EVENTS, DETECTORS, SUMMARY, RESOLUTION, REFERENCE = (
    "events.csv",
    "detectors.csv",
    "summary.csv",
    "resolution.csv",
    "reference.csv",
)
POST_HOC = "post-hoc.csv"


@dataclass(frozen=True)
class Detector:
    """One detector's scores over the windows, one array per repeat.

    Attributes:
        name: The detector and, where it has one, its pretraining.
        family: ``DEPARTURES``, ``OTHER`` or ``TAUGHT``.
        repeats: The scores of each repeat, in the windows' order.
    """

    name: str
    family: str
    repeats: tuple[Floats, ...]


@dataclass(frozen=True)
class EventFacts:
    """What the annotations say of a scored event.

    Attributes:
        kind: The benchmark's class.
        category: Anomaly or rare event.
        locality: Local or global.
        length: Point or subsequence.
        starts: The month it starts in.
        channels: The channels the corpus reads that it is annotated on.
    """

    kind: str
    category: str
    locality: str
    length: str
    starts: str
    channels: tuple[int, ...]


@dataclass(frozen=True)
class Sources:
    """Where a run's answers lie, read from a file so that a run's paths stay beside the run.

    Attributes:
        answers: The directories of answer files; the first holds the departures' answers.
        exports: Each campaign export and the campaigns to read from it.
        truncated_from: The run that records the windows with truncated history.
        departures_from: The directory of the departures' arrays.
        raw: The corpus's raw root.
    """

    answers: tuple[Path, ...]
    exports: tuple[tuple[Path, tuple[str, ...]], ...]
    truncated_from: Path
    departures_from: Path
    raw: Path

    @classmethod
    def read(cls, path: Path) -> "Sources":
        data = tomllib.loads(path.read_text())
        return cls(
            answers=tuple(Path(each) for each in data["answers"]),
            exports=tuple(
                (Path(each["path"]), tuple(each["campaigns"])) for each in data["exports"]
            ),
            truncated_from=Path(data["truncated_from"]),
            departures_from=Path(data["departures_from"]),
            raw=Path(data["raw"]),
        )


def answer_detector(stem: str) -> tuple[str, str, str, int] | None:
    """Side, detector, family and seed of an answer file; ``None`` for one not counted here.

    A fine-tuned reader counts on the validation months taught every event (``k99``) and on the
    selection months only from the initialisation, since a pretraining read those months.

    Raises:
        ValueError: If the name fits no family, so no detector is dropped unnoticed.
    """
    matched = SEEDED.search(stem)
    seed = int(matched.group("seed")) if matched else 1
    selection = bool(matched and matched.group("selection"))
    name = stem[: matched.start()] if matched else stem
    if name.startswith("select-ft@"):
        if not selection or "initialisation" not in name:
            return None
        return SELECT, name, TAUGHT, seed
    side = SELECT if name.startswith("select-") else REFIT
    if side == REFIT and not name.startswith(("refit-", "direction-", "trees-")):
        raise ValueError(f"{stem}: not an answer of either side")
    if re.search(r"-k\d+$", name) and not name.endswith("-k99"):
        return None
    if "taught-" in name or name.endswith(("-trees", "-direction")) or "ft@" in name:
        return side, name, TAUGHT, seed
    if name.startswith(("direction-", "trees-")):
        return side, name, TAUGHT, seed
    if "-local-" in name:
        return side, name, DEPARTURES, seed
    if name.endswith(("-whitened", "-farthest")):
        return side, name, OTHER, seed
    raise ValueError(f"{stem}: no family claims this answer")


def candidate_family(candidate: str) -> str | None:
    """The family of a campaign's candidate; ``None`` for one taught a few events."""
    if "events=" in candidate or "head=" in candidate:
        return None
    if candidate.startswith("channel_local"):
        return DEPARTURES
    if candidate.partition("@")[0] in FREE_CANDIDATES:
        return OTHER
    raise ValueError(f"{candidate}: no family claims this candidate")


def month_slices(units: Ints) -> list[tuple[int, int]]:
    """Where each unit's run of windows starts and how long it is.

    Raises:
        ValueError: If a unit's windows are not contiguous.
    """
    changes = np.flatnonzero(np.diff(units)) + 1
    starts = np.concatenate(([0], changes))
    lengths = np.diff(np.concatenate((starts, [len(units)])))
    if len(set(units[starts].tolist())) != len(starts):
        raise ValueError("a unit's windows are not contiguous")
    return list(zip(starts.tolist(), lengths.tolist(), strict=True))


def rotation_order(slices: Sequence[tuple[int, int]], draws: np.random.Generator) -> Ints:
    """An index that turns every month by a random offset of at least one window."""
    parts = []
    for start, length in slices:
        offset = int(draws.integers(1, length)) if length > 1 else 0
        parts.append(start + (np.arange(length) + offset) % length)
    return np.concatenate(parts)


def found_mask(scores: Floats, events: Ints, event_ids: Ints, share: float) -> Bools:
    """Which of ``event_ids`` hold a window above the alarm at ``share`` of normal windows.

    The alarm stops before the normal window that would flag more than ``share`` of them, so a
    window ties with that normal window and is not flagged, as the harness counts it.
    """
    normal = scores[events == 0]
    allowed = int(share * len(normal))
    if allowed >= len(normal):
        boundary = -np.inf
    else:
        boundary = float(np.partition(normal, len(normal) - 1 - allowed)[len(normal) - 1 - allowed])
    flagged = np.unique(events[(scores > boundary) & (events > 0)])
    return np.isin(event_ids, flagged)


def detector_found(
    detector: Detector,
    events: Ints,
    event_ids: Ints,
    share: float,
    order: Ints | None = None,
) -> Bools:
    """The events the detector finds in at least half its repeats, optionally rotated."""
    counts = np.zeros(len(event_ids), dtype=np.int64)
    for scores in detector.repeats:
        turned = scores if order is None else scores[order]
        counts += found_mask(turned, events, event_ids, share)
    needed = math.ceil(len(detector.repeats) / 2)
    found: Bools = counts >= needed
    return found


def family_found(
    detectors: Sequence[Detector],
    family: str,
    events: Ints,
    event_ids: Ints,
    share: float,
    order: Ints | None = None,
) -> Bools:
    """The events any detector of ``family`` finds."""
    found = np.zeros(len(event_ids), dtype=bool)
    for detector in detectors:
        if detector.family == family:
            found |= detector_found(detector, events, event_ids, share, order)
    return found


def categories(departures: Bools, other: Bools, taught: Bools) -> list[str]:
    """Each event under the first family that finds it."""
    named = []
    for by_departures, by_other, by_taught in zip(
        departures.tolist(), other.tolist(), taught.tolist(), strict=True
    ):
        if by_departures:
            named.append(DEPARTURES)
        elif by_other:
            named.append(OTHER)
        elif by_taught:
            named.append(TAUGHT)
        else:
            named.append(NOTHING)
    return named


@dataclass(frozen=True)
class Excess:
    """The events the departures miss that other detectors find, against chance.

    Attributes:
        missed: How many events the departures miss.
        found: How many of those the other detectors find.
        expected: The mean count under the rotations.
        upper: The rotations' 95th percentile.
        per_event: Each event's share of rotations in which the other detectors find it.
    """

    missed: int
    found: int
    expected: float
    upper: float
    per_event: Floats

    @property
    def excess(self) -> float:
        return self.found - self.expected


def excess_over_rotations(
    detectors: Sequence[Detector],
    structure: Structure,
    event_ids: Ints,
    departures: Bools,
    other: Bools,
    *,
    share: float,
    rotations: int,
    draws: np.random.Generator,
) -> Excess:
    """The count of missed events the other detectors find, and its rotations' distribution."""
    missed = ~departures
    slices = month_slices(structure.units)
    counts = np.empty(rotations, dtype=np.int64)
    per_event = np.zeros(len(event_ids), dtype=np.int64)
    for place in range(rotations):
        order = rotation_order(slices, draws)
        turned = family_found(detectors, OTHER, structure.events, event_ids, share, order)
        counts[place] = int((turned & missed).sum())
        per_event += turned
    return Excess(
        missed=int(missed.sum()),
        found=int((other & missed).sum()),
        expected=float(counts.mean()),
        upper=float(np.quantile(counts, 0.95)),
        per_event=per_event / rotations,
    )


def fused(detectors: Sequence[Detector]) -> Floats:
    """The largest of each detector's rank over all windows, its first repeat."""
    ranks = np.stack([rankdata(d.repeats[0]) / len(d.repeats[0]) for d in detectors])
    largest: Floats = ranks.max(axis=0)
    return largest


def month_referenced(scores: Floats, units: Ints) -> Floats:
    """Each score less its month's median, over the month's scaled median absolute deviation."""
    referenced = scores.copy()
    for start, length in month_slices(units):
        month = scores[start : start + length]
        centre = float(np.median(month))
        spread = 1.4826 * float(np.median(np.abs(month - centre)))
        referenced[start : start + length] = (month - centre) / (spread if spread > 0.0 else 1.0)
    return referenced


def reopens_perturbation_pretext(
    excess: Excess, *, gain: float, lower: float, event_count: int
) -> bool:
    """Whether the perturbation pretext has room over the departures.

    It has when the excess over the rotations is beyond their 95th percentile, is at least the
    events the least gain asks of one detector, and a fusion at one alarm rate beats the
    departures alone.
    """
    wanted = math.ceil(LEAST_GAIN * event_count)
    return (
        excess.found > excess.upper
        and excess.excess >= wanted
        and confirms_recall(gain, lower, minimum=FLOOR)
    )


def resolution_found(
    departures: Floats,
    events: Ints,
    event: int,
    channels: Sequence[int],
    share: float,
) -> bool:
    """Whether an annotated channel's departure stands above that channel's normal windows."""
    windows = departures[events == event]
    normal = events == 0
    for channel in channels:
        scores = departures[:, channel]
        values = scores[normal]
        allowed = int(share * len(values))
        boundary = float(np.partition(values, len(values) - 1 - allowed)[len(values) - 1 - allowed])
        if (windows[:, channel] > boundary).any():
            return True
    return False


def channel_alarms(
    departures: Floats, events: Ints, channels: Sequence[int], share: float
) -> Floats:
    """Each channel's alarm: the departure above which ``share`` of its normal windows lie."""
    normal = departures[events == 0][:, list(channels)]
    allowed = int(share * len(normal))
    place = len(normal) - 1 - allowed
    alarms: Floats = np.partition(normal, place, axis=0)[place]
    return alarms


def event_signal(values: Floats, alarms: Floats) -> float:
    """How far the largest departure stands over its channel's alarm; nought with no alarm set.

    ``values`` is ``[windows, channels]``. A channel whose alarm is nought never departs in normal
    windows, so a ratio to it says nothing and it is left out.
    """
    usable = alarms > 0.0
    if not usable.any() or not len(values):
        return 0.0
    return float((values[:, usable] / alarms[usable]).max())


def chance_of_signal(
    departures: Floats,
    events: Ints,
    event: int,
    channels: Sequence[int],
    orders: Sequence[Ints],
    alarms: Floats,
) -> float:
    """The share of rotations whose windows stand as far above the alarm as the event's own.

    The size of the departure is what is compared, since the chance of merely passing the alarm
    grows with an event's windows and channels whatever its signal.
    """
    held = np.flatnonzero(events == event)
    columns = departures[:, list(channels)]
    observed = event_signal(columns[held], alarms)
    if observed == 0.0:
        return 1.0
    return float(
        np.mean([event_signal(columns[order[held]], alarms) >= observed for order in orders])
    )


def queues_finer_cells(rows: Sequence[Mapping[str, str]]) -> bool:
    """Whether the ten-minute cells show enough events the hour does not, beyond chance."""
    gained = [row for row in rows if row["gained"] == "1"]
    return (
        len(gained) >= QUEUE_EVENTS
        and len({row["channels"] for row in gained}) >= QUEUE_CHANNEL_SETS
    )


def read_answer_csv(path: Path) -> dict[Window, tuple[int, float]]:
    with path.open(newline="") as handle:
        return {
            (row["unit"], int(row["position"])): (
                int(float(row["target"])),
                float(row["predicted"]),
            )
            for row in csv.DictReader(handle)
        }


def event_facts(root: Path) -> dict[int, EventFacts]:
    """Every scored event's annotations by its number, numbered as the ground truth numbers."""
    folder = root / MISSION
    annotated = event_annotations(root, MISSION, CHANNEL_ROWS)
    with (folder / "anomaly_types.csv").open(newline="") as handle:
        types = {row["ID"]: row for row in csv.DictReader(handle)}
    channels: dict[str, list[int]] = defaultdict(list)
    starts: dict[str, str] = {}
    with (folder / "labels.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            channel = int(row["Channel"].removeprefix("channel_"))
            if channel in CHANNEL_ROWS:
                channels[row["ID"]].append(channel)
                month = row["StartTime"][:7]
                starts[row["ID"]] = min(month, starts.get(row["ID"], month))
    numbered = [
        identifier
        for identifier in channels
        if types[identifier]["Category"] != "Communication Gap"
    ]
    facts = {}
    for number, identifier in enumerate(numbered, start=1):
        row = types[identifier]
        if annotated[number][0] != row["Class"]:
            raise SystemExit(f"event {number}: the numbering is not the ground truth's")
        facts[number] = EventFacts(
            kind=row["Class"],
            category=row["Category"],
            locality=row["Locality"],
            length=row["Length"],
            starts=starts[identifier],
            channels=tuple(sorted(set(channels[identifier]))),
        )
    return facts


def load_detectors(
    sources: Sources, side: str, windows: Sequence[Window]
) -> tuple[list[Detector], dict[str, int], dict[Window, int]]:
    """Every counted detector over ``windows``; a detector short of one of them stops the run."""
    position = {window: place for place, window in enumerate(windows)}
    grouped: dict[tuple[str, str], list[Floats]] = defaultdict(list)
    targets: dict[Window, int] = {}

    def add(name: str, family: str, answers: Mapping[Window, tuple[int, float]]) -> None:
        missing = len(position) - len(position.keys() & answers.keys())
        if missing:
            raise SystemExit(f"{name}: {missing} of the {len(position)} windows are not scored")
        scores = np.empty(len(position), dtype=np.float64)
        for window, place in position.items():
            target, score = answers[window]
            scores[place] = score
            if targets.setdefault(window, target) != target:
                raise SystemExit(f"{name}: another event at {window}")
        grouped[(name, family)].append(scores)

    for directory in sources.answers:
        for path in sorted((REPO_ROOT / directory).glob("*.csv")):
            parsed = answer_detector(path.stem)
            if parsed is None or parsed[0] != side:
                continue
            add(f"{parsed[1]}", parsed[2], read_answer_csv(path))
    if side == REFIT:
        for export, campaigns in sources.exports:
            read: dict[tuple[str, str, str], dict[Window, tuple[int, float]]] = defaultdict(dict)
            with (REPO_ROOT / export).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["campaign"] not in campaigns:
                        continue
                    read[(row["campaign"], row["candidate"], row["seed"])][
                        (row["unit"], int(row["position"]))
                    ] = (int(float(row["target"])), float(row["predicted"]))
            for (campaign, candidate, _), answers in sorted(read.items()):
                family = candidate_family(candidate)
                if family is not None:
                    add(f"{campaign[:8]}:{candidate}", family, answers)
    detectors = [
        Detector(name=name, family=family, repeats=tuple(repeats))
        for (name, family), repeats in sorted(grouped.items())
    ]
    return detectors, {d.name: len(d.repeats) for d in detectors}, targets


def windows_of(sources: Sources, side: str) -> list[Window]:
    """The windows every detector scores: an answer file's, less those with truncated history."""
    reference = "refit-local-max-hour-6h.csv" if side == REFIT else "select-local-max-hour-6h.csv"
    with (REPO_ROOT / sources.answers[0] / reference).open(newline="") as handle:
        all_windows = [(r["unit"], int(r["position"])) for r in csv.DictReader(handle)]
    dropped = read_truncated(REPO_ROOT / sources.truncated_from, side)
    return sorted(w for w in all_windows if w not in dropped)


def structure_over(windows: Sequence[Window], targets: Mapping[Window, int]) -> Structure:
    names = sorted({unit for unit, _ in windows})
    index = {name: place for place, name in enumerate(names)}
    return Structure(
        units=np.array([index[unit] for unit, _ in windows], dtype=np.int64),
        events=np.array([targets[window] for window in windows], dtype=np.int64),
        unit_count=len(names),
    )


def rotated_targets(structure: Structure, draws: np.random.Generator) -> Structure:
    """The trial's world: the events turned within months, so no event is read."""
    order = rotation_order(month_slices(structure.units), draws)
    return Structure(
        units=structure.units, events=structure.events[order], unit_count=structure.unit_count
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """The per-event table, the rotations and the fusion of one side, written as CSV only."""
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    draws = np.random.default_rng(arguments.seed)
    sources = Sources.read(arguments.sources)
    windows = windows_of(sources, arguments.side)
    detectors, repeats, targets = load_detectors(sources, arguments.side, windows)
    structure = structure_over(windows, targets)
    if arguments.trial:
        structure = rotated_targets(structure, draws)
    event_ids = np.unique(structure.events[structure.events > 0])
    facts = event_facts(REPO_ROOT / sources.raw)
    found = {
        family: family_found(detectors, family, structure.events, event_ids, SHARE)
        for family in (DEPARTURES, OTHER, TAUGHT)
    }
    named = categories(found[DEPARTURES], found[OTHER], found[TAUGHT])
    excess = excess_over_rotations(
        detectors,
        structure,
        event_ids,
        found[DEPARTURES],
        found[OTHER],
        share=SHARE,
        rotations=arguments.rotations,
        draws=draws,
    )
    fitted_before = VALIDATION_STARTS if arguments.side == REFIT else SELECTION_STARTS
    fitting_classes = {f.kind for f in facts.values() if f.starts < fitted_before}
    event_rows = [
        {
            "side": arguments.side,
            "event": event,
            "class": facts[event].kind,
            "seen": SEEN if facts[event].kind in fitting_classes else UNSEEN,
            "category_benchmark": facts[event].category,
            "locality": facts[event].locality,
            "length": facts[event].length,
            "windows": int((structure.events == event).sum()),
            "family": name,
            "other_by_chance": f"{excess.per_event[place]:.4f}",
            "unfound_before": int(event in UNFOUND_BEFORE),
        }
        for place, (event, name) in enumerate(zip(event_ids.tolist(), named, strict=True))
    ]
    once = np.ones((1, structure.unit_count))

    def mean_recall(detector: Detector) -> float:
        return float(
            np.mean([event_recalls(structure, r, once, SHARE)[0] for r in detector.repeats])
        )

    detector_rows = [
        {
            "side": arguments.side,
            "detector": d.name,
            "family": d.family,
            "repeats": repeats[d.name],
            "recall": f"{mean_recall(d):.4f}",
            "found": int(detector_found(d, structure.events, event_ids, SHARE).sum()),
        }
        for d in detectors
    ]
    summary = {
        "side": arguments.side,
        "events": len(event_ids),
        "windows": len(windows),
        "detectors": len(detectors),
        **{name: named.count(name) for name in (DEPARTURES, OTHER, TAUGHT, NOTHING)},
        "missed_by_departures": excess.missed,
        "other_finds": excess.found,
        "expected_by_chance": f"{excess.expected:.2f}",
        "upper_by_chance": f"{excess.upper:.1f}",
        "excess": f"{excess.excess:.2f}",
        "rotations": arguments.rotations,
    }
    if arguments.side == REFIT:
        by_name = {d.name: d for d in detectors}
        chosen = [by_name[f"{campaign}:{candidate}"] for campaign, candidate in REPRESENTATIVES]
        control = chosen[0].repeats[0]
        gain, lower, upper = recall_interval(
            structure,
            control,
            fused(chosen),
            share=SHARE,
            resamples=FUSION_RESAMPLES,
            draws=draws,
        )
        summary |= {
            "fusion_gain": f"{gain:.4f}",
            "fusion_low": f"{lower:.4f}",
            "fusion_high": f"{upper:.4f}",
            "reopens_perturbation_pretext": int(
                reopens_perturbation_pretext(
                    excess, gain=gain, lower=lower, event_count=len(event_ids)
                )
            ),
            "least_events": math.ceil(LEAST_GAIN * len(event_ids)),
        }
    write_csv(out / EVENTS, event_rows)
    write_csv(out / DETECTORS, detector_rows)
    write_csv(out / SUMMARY, [summary])
    print(
        f"{arguments.side}: {len(windows)} windows, {len(event_ids)} events, "
        f"{len(detectors)} detectors, {sum(repeats.values())} repeats, "
        f"{arguments.rotations} rotations; CSV written"
    )


def resolution(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """The annotated channels of the events nothing found, ten-minute cells against the hour."""
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    draws = np.random.default_rng(arguments.seed)
    sources = Sources.read(arguments.sources)
    with (REPO_ROOT / sources.answers[0] / "refit-local-max-hour-6h.csv").open(
        newline=""
    ) as handle:
        answered = [
            (r["unit"], int(r["position"]), int(float(r["target"]))) for r in csv.DictReader(handle)
        ]
    dropped = read_truncated(REPO_ROOT / sources.truncated_from, REFIT)
    kept_rows = [k for k, (u, p, _) in enumerate(answered) if (u, p) not in dropped]
    indices = FIRST_REFIT_INDEX + np.array(kept_rows, dtype=np.int64)
    targets = {(u, p): t for u, p, t in answered}
    windows = [(answered[k][0], answered[k][1]) for k in kept_rows]
    structure = structure_over(windows, targets)
    if arguments.trial:
        structure = rotated_targets(structure, draws)
    hour = np.load(REPO_ROOT / sources.departures_from / "refit-hour-6h.npy", mmap_mode="r")[
        indices
    ]
    bins = np.load(REPO_ROOT / sources.departures_from / "refit-bins-6h.npy", mmap_mode="r")[
        indices
    ]
    facts = event_facts(REPO_ROOT / sources.raw)
    rows_of = {channel: row for row, channel in enumerate(CHANNEL_ROWS)}
    slices = month_slices(structure.units)
    orders = [rotation_order(slices, draws) for _ in range(arguments.rotations)]
    rows = []
    for event in sorted(UNFOUND_BEFORE):
        if not (structure.events == event).any():
            rows.append({"event": event, "gained": "", "note": "no window is read"})
            continue
        channels = [rows_of[c] for c in facts[event].channels]
        hour_found = resolution_found(hour, structure.events, event, channels, SHARE)
        bins_found = resolution_found(bins, structure.events, event, channels, SHARE)
        hour_alarms = channel_alarms(hour, structure.events, channels, SHARE)
        bins_alarms = channel_alarms(bins, structure.events, channels, SHARE)
        hour_chance = chance_of_signal(hour, structure.events, event, channels, orders, hour_alarms)
        bins_chance = chance_of_signal(bins, structure.events, event, channels, orders, bins_alarms)
        hour_real = hour_found and hour_chance <= CHANCE_RATE
        bins_real = bins_found and bins_chance <= CHANCE_RATE
        rows.append(
            {
                "event": event,
                "class": facts[event].kind,
                "windows": int((structure.events == event).sum()),
                "channels": " ".join(str(c) for c in facts[event].channels),
                "hour_found": int(hour_found),
                "hour_chance": f"{hour_chance:.3f}",
                "bins_found": int(bins_found),
                "bins_chance": f"{bins_chance:.3f}",
                "gained": int(bins_real and not hour_real),
                "note": "",
            }
        )
    write_csv(out / RESOLUTION, rows)
    summaries = np.load(REPO_ROOT / sources.departures_from / "summaries.npy", mmap_mode="r")[
        indices
    ]
    present = ~np.isnan(np.asarray(summaries[..., 0]))
    scores = window_scores(np.asarray(hour), present, MAX)
    once = np.ones((1, structure.unit_count))
    write_csv(
        out / REFERENCE,
        [
            {
                "detector": "local max, hour-6h",
                "reference": name,
                "recall": f"{float(event_recalls(structure, values, once, SHARE)[0]):.4f}",
            }
            for name, values in (
                ("none", scores),
                ("each month", month_referenced(scores, structure.units)),
            )
        ],
    )
    print(f"{len(rows)} events, {arguments.rotations} rotations; CSV written")


def correlation(raw: Floats, values: Floats) -> float:
    """The correlation over the hours both hold; ``nan`` where too few vary."""
    present = np.isfinite(raw) & np.isfinite(values)
    if present.sum() < 10 or np.std(raw[present]) == 0.0 or np.std(values[present]) == 0.0:
        return float("nan")
    return float(np.corrcoef(raw[present], values[present])[0, 1])


def preflight(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """One month of each channel's hourly mean against the departures' rows, or the run stops."""
    import io

    import pandas as pd

    from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS

    sources = Sources.read(arguments.sources)
    with (REPO_ROOT / sources.answers[0] / "refit-local-max-hour-6h.csv").open(
        newline=""
    ) as handle:
        ended = [
            (k, float(r["ends_at"]))
            for k, r in enumerate(csv.DictReader(handle))
            if r["unit"] == f"{MISSION}/2007-03"
        ]
    indices = FIRST_REFIT_INDEX + np.array([k for k, _ in ended], dtype=np.int64)
    hours = np.array([end for _, end in ended], dtype=np.float64)
    summaries = np.load(REPO_ROOT / sources.departures_from / "summaries.npy", mmap_mode="r")[
        indices
    ]
    origin = pd.Timestamp(ORIGIN_SECONDS, unit="s")
    wrong = []
    lines = []
    means = np.array(summaries[:, :, 0])
    for row, channel in enumerate(CHANNEL_ROWS):
        path = REPO_ROOT / sources.raw / MISSION / "channels" / f"channel_{channel}.zip"
        with zipfile.ZipFile(path) as archive:
            frame = pd.read_pickle(io.BytesIO(archive.read(archive.namelist()[0])))
        series = frame.iloc[:, 0]
        ends = (series.index - origin) / pd.Timedelta(hours=1)
        slot = np.ceil(np.asarray(ends, dtype=np.float64))
        hourly = pd.Series(np.asarray(series, dtype=np.float64)).groupby(slot).mean()
        raw = hourly.reindex(hours).to_numpy()
        held = np.isfinite(raw) & np.isfinite(means[:, row])

        own = correlation(raw, means[:, row])
        others = [
            abs(correlation(raw, means[:, column]))
            for column in range(len(CHANNEL_ROWS))
            if column != row
        ]
        rivals = [other for other in others if np.isfinite(other)]
        best = max(rivals) if rivals else float("nan")
        lines.append((str(channel), str(row), f"{own:.4f}", f"{best:.4f}", str(int(held.sum()))))
        if np.isfinite(own) and (own < 0.99 or (rivals and own <= best)):
            wrong.append(channel)
    print(table(("channel", "row", "own", "best other", "hours"), lines))
    if wrong:
        raise SystemExit(f"the rows of channels {wrong} do not match the raw files")
    print("every channel with a varying mean matches its row")


def read_tables(directory: Path) -> str:
    """The tables a note pastes, from the CSV alone."""
    parts = []
    for side in (REFIT, SELECT):
        path = directory / side / SUMMARY
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(newline="")))
        parts.append(f"**{side}**\n\n" + table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
        events = list(csv.DictReader((directory / side / EVENTS).open(newline="")))
        for column, title in (
            ("category_benchmark", "benchmark category"),
            ("locality", "locality"),
            ("seen", "class on the fitting side"),
        ):
            tally: dict[tuple[str, str], int] = defaultdict(int)
            for row in events:
                tally[(row[column], row["family"])] += 1
            parts.append(
                table(
                    (title, "family", "events"),
                    [(a, b, str(n)) for (a, b), n in sorted(tally.items())],
                )
            )
    path = directory / POST_HOC
    if path.exists():
        rows = list(csv.DictReader(path.open(newline="")))
        parts.append("**post hoc**\n\n" + table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
    path = directory / REFERENCE
    if path.exists():
        rows = list(csv.DictReader(path.open(newline="")))
        parts.append(table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
    path = directory / RESOLUTION
    if path.exists():
        rows = list(csv.DictReader(path.open(newline="")))
        parts.append(table(tuple(rows[0]), [tuple(r.values()) for r in rows]))
        readable = [row for row in rows if row.get("gained")]
        parts.append(f"finer cells queued: {queues_finer_cells(readable)}")
    return "\n\n".join(parts)


def rotated_counts(
    detector_set: Sequence[Detector],
    family: str,
    structure: Structure,
    event_ids: Ints,
    orders: Sequence[Ints],
) -> Ints:
    """How many events ``family`` finds under each rotation."""
    return np.array(
        [
            int(family_found(detector_set, family, structure.events, event_ids, SHARE, o).sum())
            for o in orders
        ],
        dtype=np.int64,
    )


def fused_over_repeats(fixed: Sequence[Detector], varied: Detector) -> list[Floats]:
    """The fusion once per repeat of ``varied``, the other members at their first repeat."""
    return [
        fused([*fixed, Detector(varied.name, varied.family, (scores,))])
        for scores in varied.repeats
    ]


def post_hoc(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """What was asked after the declared reading; it decides nothing."""
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    draws = np.random.default_rng(arguments.seed)
    sources = Sources.read(arguments.sources)
    windows = windows_of(sources, REFIT)
    detectors, _, targets = load_detectors(sources, REFIT, windows)
    structure = structure_over(windows, targets)
    event_ids = np.unique(structure.events[structure.events > 0])
    by_name = {d.name: d for d in detectors}
    once = np.ones((1, structure.unit_count))
    slices = month_slices(structure.units)
    orders = [rotation_order(slices, draws) for _ in range(arguments.rotations)]

    def recall(scores: Floats) -> float:
        return float(event_recalls(structure, scores, once, SHARE)[0])

    members = [by_name[f"{campaign}:{candidate}"] for campaign, candidate in REPRESENTATIVES]
    local, farthest, forecast, interpolation, states = members
    rows: list[dict[str, str]] = []
    union = rotated_counts(detectors, DEPARTURES, structure, event_ids, orders)
    alone = rotated_counts(
        [Detector(local.name, DEPARTURES, local.repeats)],
        DEPARTURES,
        structure,
        event_ids,
        orders,
    )
    for name, chance, real in (
        (
            "departures, any arm",
            union,
            family_found(detectors, DEPARTURES, structure.events, event_ids, SHARE),
        ),
        ("channel_local", alone, detector_found(local, structure.events, event_ids, SHARE)),
    ):
        rows.append(
            {
                "kind": "chance",
                "name": name,
                "value": str(int(real.sum())),
                "low": f"{float(chance.mean()):.2f}",
                "high": f"{float(np.quantile(chance, 0.95)):.1f}",
                "beside": "events found; the rotations' mean and 95th percentile",
            }
        )
    bars = {
        "channel_local": local.repeats[0],
        "local mean, hour-6h": by_name["refit-local-mean-hour-6h"].repeats[0],
        "local mean, bins-24h": by_name["refit-local-mean-bins-24h"].repeats[0],
    }
    for bar, scores in bars.items():
        rows.append(
            {
                "kind": "bar",
                "name": bar,
                "value": f"{recall(scores):.4f}",
                "low": "",
                "high": "",
                "beside": "recall",
            }
        )
    sets = {
        "all five": members,
        "no model": [local, farthest, interpolation],
        "without the forecast": [local, farthest, interpolation, states],
        "without the states": [local, farthest, forecast, interpolation],
        "without the interpolation": [local, farthest, forecast, states],
        "without the cells": [local, forecast, interpolation, states],
        "without the departures": [farthest, forecast, interpolation, states],
    }
    for name, chosen in sets.items():
        scores = fused(chosen)
        for bar, control in bars.items():
            gain, lower, upper = recall_interval(
                structure, control, scores, share=SHARE, resamples=FUSION_RESAMPLES, draws=draws
            )
            rows.append(
                {
                    "kind": "fusion",
                    "name": name,
                    "value": f"{gain:+.4f}",
                    "low": f"{lower:+.4f}",
                    "high": f"{upper:+.4f}",
                    "beside": f"recall {recall(scores):.4f}, against {bar}",
                }
            )
    no_model = [local, farthest, interpolation]
    for name, detector in (
        ("frozen, base1", "3bed588b:frozen_knn@distance=whitened,reference=unit"),
        ("initialisation, base1", "3bed588b:untrained_knn@distance=whitened,reference=unit"),
        ("frozen, base2", "631d0f65:frozen_knn@distance=whitened,reference=unit"),
        ("initialisation, base2", "631d0f65:untrained_knn@distance=whitened,reference=unit"),
    ):
        values = [recall(scores) for scores in fused_over_repeats(no_model, by_name[detector])]
        rows.append(
            {
                "kind": "states member",
                "name": name,
                "value": f"{float(np.mean(values)):.4f}",
                "low": f"{min(values):.4f}",
                "high": f"{max(values):.4f}",
                "beside": f"recall of the no-model fusion with it, over {len(values)} repeats",
            }
        )
    for bar, scores in bars.items():
        if bar == "channel_local":
            continue
        gain, lower, upper = recall_interval(
            structure,
            bars["channel_local"],
            scores,
            share=SHARE,
            resamples=FUSION_RESAMPLES,
            draws=draws,
        )
        rows.append(
            {
                "kind": "bar gap",
                "name": bar,
                "value": f"{gain:+.4f}",
                "low": f"{lower:+.4f}",
                "high": f"{upper:+.4f}",
                "beside": "against channel_local",
            }
        )
    write_csv(out / POST_HOC, rows)
    print(f"{len(rows)} rows; CSV written")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measuring = commands.add_parser("measure")
    measuring.add_argument("--side", choices=(REFIT, SELECT), required=True)
    measuring.add_argument("--sources", type=Path, required=True)
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--rotations", type=int, default=1000)
    measuring.add_argument("--seed", type=int, default=1)
    measuring.add_argument("--trial", action="store_true")
    finer = commands.add_parser("resolution")
    finer.add_argument("--sources", type=Path, required=True)
    finer.add_argument("--out", type=Path, required=True)
    finer.add_argument("--rotations", type=int, default=1000)
    finer.add_argument("--seed", type=int, default=1)
    finer.add_argument("--trial", action="store_true")
    checking = commands.add_parser("preflight")
    checking.add_argument("--sources", type=Path, required=True)
    checking.add_argument("--out", type=Path, required=True)
    after = commands.add_parser("post-hoc")
    after.add_argument("--sources", type=Path, required=True)
    after.add_argument("--out", type=Path, required=True)
    after.add_argument("--rotations", type=int, default=1000)
    after.add_argument("--seed", type=int, default=11)
    reading = commands.add_parser("read")
    reading.add_argument("--dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
    arguments = parse_arguments(argv)
    match arguments.command:
        case "measure":
            measure(arguments)
        case "resolution":
            resolution(arguments)
        case "preflight":
            preflight(arguments)
        case "post-hoc":
            post_hoc(arguments)
        case "read":
            print(read_tables(arguments.dir))


if __name__ == "__main__":  # pragma: no cover
    main()
