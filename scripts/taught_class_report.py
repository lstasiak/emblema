"""Which classes a taught detector learnt, and how it reads their events against the rest.

A detector taught a few of the fitting side's events is handed whole events, each of a class
the benchmark names. An operator who marks incidents marks the kinds that recur, so the question
the product asks is twofold: does teaching lift the events of the classes it saw, and does it
lift, or lose, the events of the classes nobody marked? This script answers it off the answers a
campaign exported (``predictions.csv`` of ``campaign_pairs_report.py``): for every cell of a
candidate taught events it draws the taught events again as the harness drew them, from the
fitting side's scored events under the cell's seed, names their classes, splits the scored
events into those of a taught class and the rest, and reads the event-weighted area on each
half beside the untaught cell of the same seed.

    uv run scripts/taught_class_report.py --task esa-mission1-events --task-id UUID
        --predictions DIR/predictions.csv --out DIR
        --pair TAUGHT_CANDIDATE UNTAUGHT_CANDIDATE [--pair ...] [--first 2000-01 --last 2006-12]

The fitting months are the mission's first half, as the task cuts them; ``--first`` and
``--last`` state them, and the drawn events are checked against what a cell was taught where a
harness printed it. Writes ``taught-classes.csv``, one row per cell and half, and prints the
gains per candidate and half, mean and standard error over the seeds.
"""

import argparse
import csv
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from uuid import UUID

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from emblema.catalog.adapters.readers.esa_ad_raw import ORIGIN_SECONDS, EsaAdRawCorpusReader
from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.adapters.readers.esa_ad_raw_ground_truth import EsaAdRawGroundTruth
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.detection.taught_events import TaughtEvents
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.campaign_pairs_report import PredictionRow, read_predictions
from scripts.detection_power_report import months_between, window_ends_of
from scripts.raw_corpora import raw_root
from scripts.reporting import table

TAUGHT_CLASSES = "taught-classes.csv"
COLUMNS = (
    "campaign",
    "taught_candidate",
    "untaught_candidate",
    "seed",
    "events",
    "taught_classes",
    "half",
    "scored_events",
    "area_taught",
    "area_untaught",
    "gain",
)
TAUGHT_HALF, OTHER_HALF = "taught classes", "other classes"
_EVENTS = re.compile(r"(?:^|[@,])events=(\d+)(?:,|$)")


@dataclass(frozen=True)
class FittingEvents:
    """The fitting side's scored events as the harness numbers them, and every event's class."""

    pool: tuple[LabelledWindow, ...]
    classes: Mapping[int, str]

    @classmethod
    def of(cls, truth: EsaAdRawGroundTruth, mission: str, first: str, last: str) -> "FittingEvents":
        windows = [
            TaskWindow(unit=UnitKey(key), position=position, ends_at=hour)
            for key in months_between(mission, first, last)
            for position, hour in enumerate(window_ends_of(key))
        ]
        truths = truth.truths_of(KnownGroundTruths.ESA_AD_RAW, windows)
        pool = tuple(
            LabelledWindow(window=window, target=truths[window])
            for window in windows
            if AnomalyScheme.is_event(truths[window])
        )
        return cls(pool=pool, classes=truth.classes_of(mission))

    def taught_classes(self, task: TaskId, events: int, seed: int) -> frozenset[str]:
        """The classes of the events a cell under ``seed`` taught ``events`` of was handed."""
        drawn = TaughtEvents.drawn(task, self.pool, events=events, seed=seed)
        return frozenset(self.classes[number] for number in drawn.numbers())


def events_taught_by(candidate: str) -> int:
    """How many events a candidate's name says it was taught; nought where it says nothing."""
    found = _EVENTS.search(candidate)
    return int(found.group(1)) if found else 0


def halves(
    rows: Sequence[PredictionRow], classes: Mapping[int, str], taught: frozenset[str]
) -> dict[str, list[WindowPrediction]]:
    """The cell's answers in two halves, each the normal windows plus the events of some classes.

    The first half holds the events of the classes the cell was taught, the second the rest.
    """
    split: dict[str, list[WindowPrediction]] = {TAUGHT_HALF: [], OTHER_HALF: []}
    for row in rows:
        prediction = row.prediction()
        if row.target == AnomalyScheme.NORMAL:
            split[TAUGHT_HALF].append(prediction)
            split[OTHER_HALF].append(prediction)
        else:
            half = TAUGHT_HALF if classes[int(row.target)] in taught else OTHER_HALF
            split[half].append(prediction)
    return split


def read(
    predictions: Sequence[PredictionRow],
    fitting: FittingEvents,
    task: TaskId,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """One row per taught cell and half: the area taught and untaught, and the gain."""
    by_cell: dict[tuple[str, str, int], list[PredictionRow]] = {}
    for row in predictions:
        by_cell.setdefault((row.campaign, row.candidate, row.seed), []).append(row)
    out: list[dict[str, str]] = []
    for taught_name, untaught_name in pairs:
        events = events_taught_by(taught_name)
        if not events:
            raise SystemExit(f"{taught_name} names no events it was taught")
        cells = sorted(key for key in by_cell if key[1] == taught_name)
        if not cells:
            raise SystemExit(f"no cell of {taught_name} among the exported answers")
        for campaign, _, seed in cells:
            untaught = by_cell.get((campaign, untaught_name, seed))
            if untaught is None:
                raise SystemExit(f"no cell of {untaught_name} under seed {seed} in {campaign}")
            classes = fitting.taught_classes(task, events, seed)
            taught_halves = halves(by_cell[(campaign, taught_name, seed)], fitting.classes, classes)
            untaught_halves = halves(untaught, fitting.classes, classes)
            for half in (TAUGHT_HALF, OTHER_HALF):
                scored = {int(p.target) for p in taught_halves[half] if p.target}
                if not scored:
                    continue
                area_taught = AnomalyRanking.of(taught_halves[half]).average_precision
                area_untaught = AnomalyRanking.of(untaught_halves[half]).average_precision
                out.append(
                    {
                        "campaign": campaign,
                        "taught_candidate": taught_name,
                        "untaught_candidate": untaught_name,
                        "seed": str(seed),
                        "events": str(events),
                        "taught_classes": " ".join(sorted(classes)),
                        "half": half,
                        "scored_events": str(len(scored)),
                        "area_taught": repr(area_taught),
                        "area_untaught": repr(area_untaught),
                        "gain": repr(area_taught - area_untaught),
                    }
                )
    return out


def render(rows: Sequence[Mapping[str, str]]) -> str:
    """The gains per campaign, candidate and half: mean and standard error over the seeds."""
    grouped: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["campaign"], row["taught_candidate"], row["half"]), []).append(row)
    lines = []
    for (campaign, candidate, half), found in sorted(grouped.items()):
        gains = [float(row["gain"]) for row in found]
        se = stdev(gains) / len(gains) ** 0.5 if len(gains) > 1 else 0.0
        lines.append(
            (
                campaign[:8],
                candidate,
                half,
                f"{mean(int(row['scored_events']) for row in found):.1f}",
                f"{mean(float(row['area_untaught']) for row in found):.3f}",
                f"{mean(float(row['area_taught']) for row in found):.3f}",
                f"{mean(gains):+.3f} ± {se:.3f}",
                str(len(found)),
            )
        )
    return table(
        ("campaign", "candidate", "half", "events", "untaught", "taught", "gain ± SE", "seeds"),
        lines,
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-id", required=True, help="the task's identity in the registry")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        nargs=2,
        metavar=("TAUGHT", "UNTAUGHT"),
        required=True,
        help="a taught candidate and the untaught one it is read beside; repeatable",
    )
    parser.add_argument("--first", default="2000-01", help="first fitting month")
    parser.add_argument("--last", default="2006-12", help="last fitting month")
    parser.add_argument("--raw", type=Path, default=None, help="the raw corpora's root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    task = KnownTasks.named(arguments.task)
    if task.corpus != KnownGroundTruths.ESA_AD_RAW:
        raise SystemExit(f"{arguments.task} is not a task over the satellite telemetry")
    mission = task.unit_prefix.rstrip("/")
    root = arguments.raw if arguments.raw is not None else raw_root("esa_ad")
    if root is None:
        raise SystemExit("the ESA Anomaly Dataset is not downloaded")
    truth = EsaAdRawGroundTruth(
        root,
        channels=EsaAdRawCorpusReader.POLICY.channels,
        origin_seconds=ORIGIN_SECONDS,
        window_hours=KnownGroundTruths.ESA_AD_RAW_WINDOW_HOURS,
    )
    fitting = FittingEvents.of(truth, mission, arguments.first, arguments.last)
    print(f"{len(fitting.pool)} fitting windows in {len(fitting.classes)} scored events")
    rows = read(
        read_predictions(arguments.predictions),
        fitting,
        TaskId(UUID(arguments.task_id)),
        [tuple(pair) for pair in arguments.pair],
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / TAUGHT_CLASSES).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(render(rows))


if __name__ == "__main__":
    main()
