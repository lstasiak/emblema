"""A forecasting pretext's error as a detector, read on the fitting side's later months first.

Before a reading of the pretext goes to a card, its knobs are read where no validation month is
spent: the task's fitting side is cut at a month, the earlier months fit each detector and the
later ones are scored, every event with windows on both sides left unscored, as the supervised
reference and the channel-local report cut it. Each candidate the command line names is fitted
by the package's own runtime — the pretext's error under the backbone given, or the departures
with no model — and read by the registered area and the events found with at most a stated
share of normal windows flagged, over all the scored months and over those the backbone's
pretraining never read, since the months it read are not what the validation months are.

Measurement and reading are two steps. ``score`` fits and scores: every candidate's answer for
every scored window goes to a CSV of its own beside one row of readings per candidate. ``read``
takes those CSVs and puts an interval on each candidate's readings and on the first candidate's
lead over every other, from resamples of the scored months; nothing is drawn here.

    uv run scripts/history_pretext_reading_pilot.py score --task esa-mission1-events
        --manifest KEY CHECKSUM --weights KEY CHECKSUM --select-from 2004-01
        --unread-from 2006-10 --candidate NAME [--candidate NAME ...] [--share 0.01]
        [--device mps] --out DIR
    uv run scripts/history_pretext_reading_pilot.py read --dir DIR [--resamples 2000]
        [--seed 1] [--share 0.01]
"""

import argparse
import contextlib
import csv
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.exceptions import InvalidAnomalyRankingError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.unit_counts import UnitCounts
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.reporting import table
from scripts.telemetry_supervised_reference import month_of, time_sides

READINGS = "readings.csv"
INTERVALS = "intervals.csv"
READING_COLUMNS = (
    "candidate",
    "months",
    "windows",
    "events",
    "area",
    "event_recall",
    "seconds",
)
PREDICTION_COLUMNS = ("candidate", "unit", "position", "ends_at", "target", "predicted")
INTERVAL_COLUMNS = ("reading", "measure", "point", "low", "high")
MEASURES = ("area", "event_recall")


@dataclass(frozen=True, kw_only=True)
class Reading:
    """What one candidate read on one set of months.

    Attributes:
        candidate: The detector's name, knobs and all.
        months: Which months were read: every scored month, or those the pretraining never read.
        windows: How many windows that was.
        events: How many events lay among them.
        area: The registered event-weighted area under the precision-recall curve.
        event_recall: The share of events found with at most the stated share of normal
            windows flagged.
        seconds: What the fit and the scoring took, the same for both sets of months.
    """

    candidate: str
    months: str
    windows: int
    events: int
    area: float
    event_recall: float
    seconds: float


@dataclass(frozen=True, kw_only=True)
class Interval:
    """One reading over every scored month, with its interval over resamples of the months.

    Attributes:
        reading: The candidate read, or ``<arm> - <candidate>`` for the first candidate's lead
            over another on the same resamples.
        measure: ``area`` or ``event_recall``.
        point: The reading with every month counted once.
        low: The 2.5th percentile over the resamples.
        high: The 97.5th percentile over the resamples.
    """

    reading: str
    measure: str
    point: float
    low: float
    high: float


def readings_of(
    candidate: str,
    predictions: Sequence[WindowPrediction],
    *,
    unread_from: str,
    share: float,
    seconds: float,
) -> list[Reading]:
    """The candidate's readings over every scored month and over the months from ``unread_from``.

    The second reading is left out where those months hold no event or no normal window, since
    an area needs both.
    """
    rows = [_reading(candidate, "all", predictions, share, seconds)]
    unread = [p for p in predictions if month_of(p) >= unread_from]
    with contextlib.suppress(InvalidAnomalyRankingError):
        rows.append(_reading(candidate, "unread", unread, share, seconds))
    return rows


def _reading(
    candidate: str,
    months: str,
    predictions: Sequence[WindowPrediction],
    share: float,
    seconds: float,
) -> Reading:
    ranking = AnomalyRanking.of(predictions)
    return Reading(
        candidate=candidate,
        months=months,
        windows=len(predictions),
        events=ranking.event_count,
        area=ranking.average_precision,
        event_recall=ranking.event_recall_at_false_alarm_share(share),
        seconds=seconds,
    )


def intervals_of(
    rankings: Mapping[str, AnomalyRanking], *, resamples: int, seed: int, share: float
) -> list[Interval]:
    """Each candidate's readings and the first candidate's lead over every other, with intervals.

    The months are drawn with replacement ``resamples`` times, the same draws for every
    candidate, so a lead is read on paired resamples.

    Raises:
        ValueError: If a candidate scored other months than the first.
    """
    names = list(rankings)
    arm = rankings[names[0]]
    units = list(arm.units)
    for name, ranking in rankings.items():
        if list(ranking.units) != units:
            raise ValueError(f"{name} scored other months than {names[0]}")
    picks = np.random.default_rng(seed).integers(0, len(units), size=(resamples, len(units)))
    counts = UnitCounts.of_picks(picks.tolist(), len(units))
    points = {
        name: (ranking.average_precision, ranking.event_recall_at_false_alarm_share(share))
        for name, ranking in rankings.items()
    }
    drawn = {
        name: (
            np.asarray(ranking.average_precisions(counts)),
            np.asarray(ranking.event_recalls(counts, share)),
        )
        for name, ranking in rankings.items()
    }
    rows = [
        _interval(name, measure, points[name][index], drawn[name][index])
        for name in names
        for index, measure in enumerate(MEASURES)
    ]
    rows.extend(
        _interval(
            f"{names[0]} - {name}",
            measure,
            points[names[0]][index] - points[name][index],
            drawn[names[0]][index] - drawn[name][index],
        )
        for name in names[1:]
        for index, measure in enumerate(MEASURES)
    )
    return rows


def _interval(reading: str, measure: str, point: float, values: np.ndarray) -> Interval:
    low, high = np.percentile(values, [2.5, 97.5])
    return Interval(reading=reading, measure=measure, point=point, low=float(low), high=float(high))


def read(directory: Path, *, resamples: int, seed: int, share: float) -> list[Interval]:
    """The intervals over the CSVs a ``score`` run left in ``directory``, written beside them.

    The candidates are read in the order the run scored them; the first is the arm.

    Raises:
        ValueError: If a CSV names another candidate than its file, or scored other months.
    """
    rankings: dict[str, AnomalyRanking] = {}
    for name in candidates_read(directory / READINGS):
        candidate, predictions = read_predictions(directory / f"{file_name_of(name)}.csv")
        if candidate != name:
            raise ValueError(f"{file_name_of(name)}.csv holds {candidate}, not {name}")
        rankings[name] = AnomalyRanking.of(predictions)
    intervals = intervals_of(rankings, resamples=resamples, seed=seed, share=share)
    write_intervals(directory / INTERVALS, intervals)
    return intervals


def candidates_read(path: Path) -> list[str]:
    """The candidates a readings file holds, each once, in the order they were scored."""
    with path.open(newline="") as handle:
        return list(dict.fromkeys(row["candidate"] for row in csv.DictReader(handle)))


def read_predictions(path: Path) -> tuple[str, list[WindowPrediction]]:
    """The candidate and the predictions a ``score`` run wrote to ``path``.

    Raises:
        ValueError: If the file holds no prediction or more than one candidate.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = {row["candidate"] for row in rows}
    if len(candidates) != 1:
        raise ValueError(f"{path} holds {len(candidates)} candidates, not one")
    return candidates.pop(), [
        WindowPrediction(
            window=TaskWindow(
                unit=UnitKey(row["unit"]),
                position=int(row["position"]),
                ends_at=float(row["ends_at"]),
            ),
            target=float(row["target"]),
            predicted=float(row["predicted"]),
        )
        for row in rows
    ]


def reading_table(readings: Iterable[Reading]) -> str:
    return table(
        ["Candidate", "Months", "Windows", "Events", "Area", "Events found", "Seconds"],
        [
            [
                r.candidate,
                r.months,
                str(r.windows),
                str(r.events),
                f"{r.area:.3f}",
                f"{r.event_recall:.3f}",
                f"{r.seconds:.0f}",
            ]
            for r in readings
        ],
    )


def interval_table(intervals: Iterable[Interval]) -> str:
    return table(
        ["Reading", "Measure", "Point", "Interval"],
        [
            [i.reading, i.measure, f"{i.point:.3f}", f"[{i.low:.3f}; {i.high:.3f}]"]
            for i in intervals
        ],
    )


def write_readings(path: Path, readings: Iterable[Reading]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(READING_COLUMNS)
        for r in readings:
            writer.writerow(
                (
                    r.candidate,
                    r.months,
                    r.windows,
                    r.events,
                    f"{r.area:.6f}",
                    f"{r.event_recall:.6f}",
                    f"{r.seconds:.1f}",
                )
            )


def write_intervals(path: Path, intervals: Iterable[Interval]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INTERVAL_COLUMNS)
        for i in intervals:
            writer.writerow(
                (i.reading, i.measure, f"{i.point:.6f}", f"{i.low:.6f}", f"{i.high:.6f}")
            )


def write_predictions(path: Path, candidate: str, predictions: Iterable[WindowPrediction]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for p in predictions:
            writer.writerow(
                (
                    candidate,
                    p.window.unit,
                    p.window.position,
                    p.window.ends_at,
                    p.target,
                    p.predicted,
                )
            )


def file_name_of(candidate: str) -> str:
    """A candidate's name as a file name: the knobs' separators replaced."""
    return candidate.replace("@", "--").replace(",", "+").replace("=", "-")


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="fit and score every candidate named")
    score.add_argument("--task", required=True, help="the task's name in the catalogue")
    score.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    score.add_argument(
        "--weights", nargs=2, required=True, metavar=("KEY", "CHECKSUM"), help="the backbone"
    )
    score.add_argument("--select-from", required=True, help="first month scored, YYYY-MM")
    score.add_argument(
        "--unread-from", required=True, help="first month the pretraining did not read, YYYY-MM"
    )
    score.add_argument("--candidate", action="append", required=True, help="a detector's name")
    score.add_argument("--share", type=float, default=0.01)
    score.add_argument("--device", default="mps")
    score.add_argument("--out", type=Path, required=True)
    reading = commands.add_parser("read", help="intervals over a score run's CSVs")
    reading.add_argument("--dir", type=Path, required=True, help="a score run's --out")
    reading.add_argument("--resamples", type=int, default=2000)
    reading.add_argument("--seed", type=int, default=1)
    reading.add_argument("--share", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - the environment
    arguments = parse_arguments(argv)
    if arguments.command == "read":
        intervals = read(
            arguments.dir,
            resamples=arguments.resamples,
            seed=arguments.seed,
            share=arguments.share,
        )
        print(interval_table(intervals))
        return
    from emblema.config.settings import Settings
    from emblema.entrypoints.cli.campaign.known_tasks import KnownTasks
    from emblema.entrypoints.configured import configured_store
    from emblema.entrypoints.known_ground_truths import KnownGroundTruths
    from emblema.entrypoints.restored_backbones import RestoredBackbones
    from emblema.entrypoints.workers.known_detectors import KnownDetectors
    from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.detection.channel_local_detection_runtime import (
        ChannelLocalDetectionRuntime,
    )
    from emblema.evaluation.adapters.detection.pretext_error_detection_runtime import (
        PretextErrorDetectionRuntime,
    )
    from emblema.evaluation.adapters.in_memory.downstream_task_repository import (
        InMemoryDownstreamTaskRepository,
    )
    from emblema.evaluation.application.use_cases.define_downstream_task import (
        DefineDownstreamTask,
    )
    from emblema.evaluation.contracts.identifiers import CandidateRef
    from emblema.evaluation.ports.detection_runtime import DetectionRuntime
    from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum

    settings = Settings()
    store = configured_store(settings)
    blocks = PublishedCorpusBlocks(store, settings.require_worker().workspace)
    corpus = BlockCorpusWindows(blocks)
    manifest = ArtifactRef(arguments.manifest[0], Checksum.parse(arguments.manifest[1]))
    tasks = InMemoryDownstreamTaskRepository()
    defined = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        KnownTasks.named(arguments.task).defined_over(manifest, corpus.describe(manifest))
    )
    task = tasks.get(defined)
    truth = KnownGroundTruths.under(Path("data/raw"))
    fitting = task.read_windows(corpus.windows_of(task.manifest, task.tuning_units))
    pool = task.labelled(fitting, truth.truths_of(task.corpus, fitting))
    cut = time_sides(pool, select_from=arguments.select_from)
    print(
        f"{len(cut.teach)} windows fit, {len(cut.read)} scored over {cut.events_read} events; "
        f"{cut.events_dropped} events on both sides of {arguments.select_from} left unscored; "
        f"months from {arguments.unread_from} unread by the pretraining: "
        f"{sum(1 for w in cut.read if month_of(w) >= arguments.unread_from)} windows",
        flush=True,
    )
    weights = ArtifactRef(arguments.weights[0], Checksum.parse(arguments.weights[1]))
    restored = RestoredBackbones(store, weights)
    pretext = PretextErrorDetectionRuntime(restored, blocks, device=arguments.device)
    local = ChannelLocalDetectionRuntime(blocks)
    catalogue = KnownDetectors.catalogue(weights)
    arguments.out.mkdir(parents=True, exist_ok=True)
    teach = [w.window for w in cut.teach]
    readings: list[Reading] = []
    for name in arguments.candidate:
        recipe = catalogue.arm_of(CandidateRef(name)).recipe(seed=1)
        runtime: DetectionRuntime = local if recipe.features.reads_channel_history else pretext
        started = time.perf_counter()
        outcome = runtime.fit(recipe, task, teach, cut.read, retain=False)
        seconds = time.perf_counter() - started
        write_predictions(arguments.out / f"{file_name_of(name)}.csv", name, outcome.predictions)
        readings.extend(
            readings_of(
                name,
                outcome.predictions,
                unread_from=arguments.unread_from,
                share=arguments.share,
                seconds=seconds,
            )
        )
        print(f"{name}: {len(outcome.predictions)} windows scored in {seconds:.0f} s", flush=True)
    write_readings(arguments.out / READINGS, readings)
    print(reading_table(readings))


if __name__ == "__main__":
    main()
