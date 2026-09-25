"""Read the stored cells of the grid and say what the curve shows, by the registered rules.

The transfer report stores a grid cell by cell; this reads one or more of those directories —
the shards of a grid run on two accelerators, say — and turns them into the three files a note
and a figure need: one row per cell and repeat with its error and the readings beside it, one
row per comparison against the control arm with its paired interval, and the trivial
predictors read off the same windows. The arithmetic every verdict rests on is the Evaluation
context's; what is here is composition and the shape of a note.

    uv run scripts/label_curve_report.py data/report/transfer/<shard> [<shard> ...] --out DIR
    uv run scripts/label_curve_report.py --report-only DIR

Every number is validation, not test, and the result is preliminary: the evaluation harness
repeats the comparison once it exists.
"""

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import NamedTuple, Self

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.entrypoints.cli.campaign.known_tasks import KnownTask, KnownTasks
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.candidate_comparison import CandidateComparison
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.bootstrap_interval import BootstrapInterval
from emblema.evaluation.domain.statistics.comparison_rules import ComparisonRules
from emblema.evaluation.domain.statistics.comparison_verdict import ComparisonVerdict
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.holm_correction import HolmCorrection
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.domain.transfer.remaining_life_metrics import RemainingLifeMetrics
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from scripts.reporting import dated_heading, table
from scripts.transfer_grid import BUDGETS, MODES, PLAN_COLUMNS, SEEDS, Stored, budget_rank

CURVE, COMPARISONS, BASELINES = "curve.csv", "comparisons.csv", "baselines.csv"
CONTROL = TransferMode.FROM_SCRATCH
# The registered endpoint: full fine-tuning against the control at 200 labelled windows. The
# other cells compared are the secondary family, whatever part of it has run.
PRIMARY = (TransferMode.FULL_FINE_TUNING, "200")
COMPARED_CELLS = (len(MODES) - 1) * len(BUDGETS)
RULES = ComparisonRules(
    minimum_relative_reduction=0.10,
    floor_share=0.03,
    correction=HolmCorrection(alpha=0.05),
    secondary_family_size=COMPARED_CELLS - 1,
)
RESAMPLES = 10_000


@dataclass(frozen=True, kw_only=True)
class CurvePoint:
    """One cell under one seed: what it cost, what it scored, and the readings beside the score.

    Attributes:
        mode: The transfer mode.
        budget: The budget of labels, as the grid names it.
        windows: The budget resolved to a count of labelled windows.
        seed: The seed of the draw and the run.
        engines: How many engines the labels came from.
        trainable_parameters: How many weights the run could change.
        rmse: The endpoint error over every validation window.
        rmse_below_ceiling: The error over windows labelled below the ceiling.
        last_window_rmse: The error over the last window of each engine.
        alpha_lambda_accuracy: Share of answers within a fifth of the label.
        asymmetric_score: The benchmark's score, a mean per window.
        seconds: What the run took.
    """

    mode: str
    budget: str
    windows: int
    seed: int
    engines: int
    trainable_parameters: int
    rmse: float
    rmse_below_ceiling: float | None
    last_window_rmse: float | None
    alpha_lambda_accuracy: float | None
    asymmetric_score: float | None
    seconds: float

    COLUMNS = (
        "mode",
        "budget",
        "windows",
        "seed",
        "engines",
        "trainable_parameters",
        "rmse",
        "rmse_below_ceiling",
        "last_window_rmse",
        "alpha_lambda_accuracy",
        "asymmetric_score",
        "seconds",
    )

    def record(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "budget": self.budget,
            "windows": str(self.windows),
            "seed": str(self.seed),
            "engines": str(self.engines),
            "trainable_parameters": str(self.trainable_parameters),
            "rmse": repr(self.rmse),
            "rmse_below_ceiling": _text(self.rmse_below_ceiling),
            "last_window_rmse": _text(self.last_window_rmse),
            "alpha_lambda_accuracy": _text(self.alpha_lambda_accuracy),
            "asymmetric_score": _text(self.asymmetric_score),
            "seconds": repr(self.seconds),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            mode=record["mode"],
            budget=record["budget"],
            windows=int(record["windows"]),
            seed=int(record["seed"]),
            engines=int(record["engines"]),
            trainable_parameters=int(record["trainable_parameters"]),
            rmse=float(record["rmse"]),
            rmse_below_ceiling=_number(record["rmse_below_ceiling"]),
            last_window_rmse=_number(record["last_window_rmse"]),
            alpha_lambda_accuracy=_number(record["alpha_lambda_accuracy"]),
            asymmetric_score=_number(record["asymmetric_score"]),
            seconds=float(record["seconds"]),
        )


def _text(value: float | None) -> str:
    return "" if value is None else repr(value)


def _number(text: str) -> float | None:
    return None if text == "" else float(text)


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """One cell against the control arm at the same budget, over the same engines and seeds.

    Attributes:
        mode: The transfer mode compared.
        budget: The budget of labels.
        seeds: How many repeats of each arm were pooled.
        control_rmse: The control's error, pooled over the repeats.
        candidate_rmse: The candidate's error, pooled over the repeats.
        control_sd: Spread of the control's error over its repeats.
        candidate_sd: Spread of the candidate's error over its repeats.
        reduction: How much lower the candidate's error is.
        relative_reduction: The reduction as a share of the control's error.
        low: Lower end of the paired interval over engines.
        high: Upper end of the paired interval over engines.
        p_value: Two-sided bootstrap p-value of the reduction.
        floor: The practical floor at this budget.
        primary: Whether this is the registered endpoint.
        rejected: Whether zero is rejected — for the endpoint, by its confirmation; for a
            secondary cell, under the Holm correction over the registered family.
        verdict: What the registered rules say about the cell, in the note's words.
    """

    mode: str
    budget: str
    seeds: int
    control_rmse: float
    candidate_rmse: float
    control_sd: float
    candidate_sd: float
    reduction: float
    relative_reduction: float
    low: float
    high: float
    p_value: float
    floor: float
    primary: bool
    rejected: bool
    verdict: str

    COLUMNS = (
        "mode",
        "budget",
        "seeds",
        "control_rmse",
        "candidate_rmse",
        "control_sd",
        "candidate_sd",
        "reduction",
        "relative_reduction",
        "low",
        "high",
        "p_value",
        "floor",
        "primary",
        "rejected",
        "verdict",
    )

    def record(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "budget": self.budget,
            "seeds": str(self.seeds),
            "control_rmse": repr(self.control_rmse),
            "candidate_rmse": repr(self.candidate_rmse),
            "control_sd": repr(self.control_sd),
            "candidate_sd": repr(self.candidate_sd),
            "reduction": repr(self.reduction),
            "relative_reduction": repr(self.relative_reduction),
            "low": repr(self.low),
            "high": repr(self.high),
            "p_value": repr(self.p_value),
            "floor": repr(self.floor),
            "primary": "yes" if self.primary else "no",
            "rejected": "yes" if self.rejected else "no",
            "verdict": self.verdict,
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            mode=record["mode"],
            budget=record["budget"],
            seeds=int(record["seeds"]),
            control_rmse=float(record["control_rmse"]),
            candidate_rmse=float(record["candidate_rmse"]),
            control_sd=float(record["control_sd"]),
            candidate_sd=float(record["candidate_sd"]),
            reduction=float(record["reduction"]),
            relative_reduction=float(record["relative_reduction"]),
            low=float(record["low"]),
            high=float(record["high"]),
            p_value=float(record["p_value"]),
            floor=float(record["floor"]),
            primary=record["primary"] == "yes",
            rejected=record["rejected"] == "yes",
            verdict=record["verdict"],
        )


@dataclass(frozen=True, kw_only=True)
class Baseline:
    """A trivial predictor scored on the validation windows: what any candidate has to beat.

    Attributes:
        name: Which predictor.
        rmse: Its error over every validation window.
    """

    name: str
    rmse: float

    COLUMNS = ("name", "rmse")

    def record(self) -> dict[str, str]:
        return {"name": self.name, "rmse": repr(self.rmse)}

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(name=record["name"], rmse=float(record["rmse"]))


@dataclass(frozen=True, kw_only=True)
class Curve:
    """The three files of a report, read or about to be written."""

    points: tuple[CurvePoint, ...]
    comparisons: tuple[Comparison, ...]
    baselines: tuple[Baseline, ...]


CellKey = tuple[str, str, int]


def predictions_of(shards: Sequence[Stored]) -> dict[CellKey, tuple[WindowPrediction, ...]]:
    """Every stored prediction of a cell held whole, by the cell it belongs to, over the shards.

    Raises:
        SystemExit: If two shards hold the same cell.
    """
    cells: dict[CellKey, list[WindowPrediction]] = {}
    seen: set[CellKey] = set()
    for shard in shards:
        held: set[CellKey] = set()
        for row in shard.predictions():
            key = (row["mode"], row["budget"], int(row["seed"]))
            if key in seen:
                raise SystemExit(f"{shard.directory} repeats a cell another shard holds: {key}")
            held.add(key)
            cells.setdefault(key, []).append(
                WindowPrediction(
                    window=TaskWindow(
                        unit=UnitKey(row["unit"]),
                        position=int(row["position"]),
                        ends_at=float(row["ends_at"]),
                    ),
                    target=float(row["target"]),
                    predicted=float(row["predicted"]),
                )
            )
        seen |= held
    return {key: tuple(rows) for key, rows in cells.items()}


def require_one_configuration(shards: Sequence[Stored], task: KnownTask) -> None:
    """Refuse shards that do not make one grid.

    One task, one commit, one backbone, one plan per mode.

    Two shards run on two accelerators differ in their device and may in what torch reports;
    they must not differ in what was learnt from, how, or under which code.

    Raises:
        SystemExit: If the runs disagree.
    """
    runs = [run for shard in shards for run in shard.runs()]
    commits = sorted({run["commit"] for run in runs})
    if len(commits) > 1:
        raise SystemExit(f"the shards were run at more than one commit: {', '.join(commits)}")
    backbones = sorted({run["backbone"] for run in runs if run["backbone"]})
    if len(backbones) > 1:
        raise SystemExit(f"the shards adapt more than one backbone: {', '.join(backbones)}")
    # A grid stored before the task was a column is the first task's.
    tasks = sorted({run.get("task") or KnownTasks.default().name for run in runs})
    if tasks != [task.name]:
        raise SystemExit(
            f"the shards hold the task(s) {', '.join(tasks)}, not {task.name}; pass --task"
        )
    settings = [name for name in PLAN_COLUMNS if name not in ("mode", "run_seed")]
    for mode in MODES:
        plans = {
            tuple(run.get(name, "") for name in settings) for run in runs if run["mode"] == mode
        }
        if len(plans) > 1:
            differing = sorted(
                name
                for index, name in enumerate(settings)
                if len({plan[index] for plan in plans}) > 1
            )
            raise SystemExit(
                f"{mode} was run under more than one plan; differing: {', '.join(differing)}"
            )


def points_of(
    shards: Sequence[Stored], cells: Mapping[CellKey, tuple[WindowPrediction, ...]], task: KnownTask
) -> tuple[CurvePoint, ...]:
    """One point per stored run, its readings computed from its predictions."""
    points = []
    for shard in shards:
        for run in shard.runs():
            key = (run["mode"], run["budget"], int(run["sample_seed"]))
            points.append(
                CurvePoint(
                    mode=run["mode"],
                    budget=run["budget"],
                    windows=int(run["windows"]),
                    seed=int(run["sample_seed"]),
                    engines=int(run["engines"]),
                    trainable_parameters=int(run["trainable_parameters"]),
                    **readings_of(cells[key], task)._asdict(),
                    seconds=float(run["seconds"]),
                )
            )
    return tuple(sorted(points, key=lambda p: (budget_rank(p.budget), MODES.index(p.mode), p.seed)))


class Readings(NamedTuple):
    """What a cell's predictions read as under a task's scheme; blank where the scheme has none."""

    rmse: float
    rmse_below_ceiling: float | None
    last_window_rmse: float | None
    alpha_lambda_accuracy: float | None
    asymmetric_score: float | None


def readings_of(predictions: Sequence[WindowPrediction], task: KnownTask) -> Readings:
    """The readings a cell's predictions give under the task's scheme.

    The endpoint is the RMSE whatever the scheme; the readings beside it are the remaining-life
    task's own — a forecast has no ceiling, no last window at the end of a life and no
    benchmark score — and are left blank for any other.
    """
    match task.labels:
        case RemainingLifeScheme():
            metrics = RemainingLifeMetrics.of(predictions, ceiling=task.labels.ceiling)
            return Readings(
                rmse=metrics.rmse,
                rmse_below_ceiling=metrics.rmse_below_ceiling,
                last_window_rmse=metrics.last_window_rmse,
                alpha_lambda_accuracy=metrics.alpha_lambda_accuracy,
                asymmetric_score=metrics.asymmetric_score,
            )
        case ForecastScheme():
            return Readings(rmse_of(predictions), None, None, None, None)


def rmse_of(predictions: Sequence[WindowPrediction]) -> float:
    return sqrt(sum((p.predicted - p.target) ** 2 for p in predictions) / len(predictions))


def comparisons_of(
    cells: Mapping[CellKey, tuple[WindowPrediction, ...]],
    task: KnownTask,
    bootstrap: PairedUnitBootstrap,
    rules: ComparisonRules,
) -> tuple[Comparison, ...]:
    """Every mode but the control against the control, at every budget both have a seed at.

    The repeats of a cell pool per engine before the pair is made; the seeds an arm holds and the
    other does not are left out, so both sides of a pair are the same seeds over the same labels.
    The endpoint is judged on its own; the rest are judged as the registered family, the cells
    that have not run counting against those that have.
    """
    budgets = sorted({budget for _, budget, _ in cells}, key=budget_rank)
    found: list[tuple[Comparison, PairedDifference, PracticalFloor]] = []
    for budget in budgets:
        for mode in MODES:
            if mode == str(CONTROL):
                continue
            seeds = sorted(
                {
                    seed
                    for stated, held, seed in cells
                    if stated == mode and held == budget and (str(CONTROL), budget, seed) in cells
                }
            )
            if not seeds:
                continue
            control_runs = [cells[str(CONTROL), budget, seed] for seed in seeds]
            candidate_runs = [cells[mode, budget, seed] for seed in seeds]
            paired = PairedUnitErrors.pooled(
                control=[UnitError.per_unit(run) for run in control_runs],
                candidate=[UnitError.per_unit(run) for run in candidate_runs],
            )
            control_error = ErrorOverRepeats.of(
                paired.rmse_control, [rmse_of(run) for run in control_runs]
            )
            candidate_error = ErrorOverRepeats.of(
                paired.rmse_candidate, [rmse_of(run) for run in candidate_runs]
            )
            difference = bootstrap.compare(paired)
            floor = rules.floor_of(control_error)
            primary = (TransferMode(mode), budget) == PRIMARY
            verdict = rules.endpoint_verdict(difference, floor) if primary else None
            found.append(
                (
                    Comparison(
                        mode=mode,
                        budget=budget,
                        seeds=len(seeds),
                        control_rmse=control_error.pooled,
                        candidate_rmse=candidate_error.pooled,
                        control_sd=control_error.spread,
                        candidate_sd=candidate_error.spread,
                        reduction=difference.reduction,
                        relative_reduction=difference.relative_reduction,
                        low=difference.interval.low,
                        high=difference.interval.high,
                        p_value=difference.p_value,
                        floor=floor.value,
                        primary=primary,
                        rejected=verdict is ComparisonVerdict.CONFIRMED,
                        verdict="" if verdict is None else str(verdict),
                    ),
                    difference,
                    floor,
                )
            )
    secondary = [index for index, (row, _, _) in enumerate(found) if not row.primary]
    if secondary:
        rejections = rules.secondary_rejections([found[index][1].p_value for index in secondary])
        for index, rejected in zip(secondary, rejections, strict=True):
            row, difference, floor = found[index]
            verdict = rules.secondary_verdict(difference, floor, rejected=rejected)
            found[index] = (
                replace(row, rejected=rejected, verdict=str(verdict)),
                difference,
                floor,
            )
    return tuple(row for row, _, _ in found)


def _spread(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def baselines_of(
    cells: Mapping[CellKey, tuple[WindowPrediction, ...]], task: KnownTask
) -> tuple[Baseline, ...]:
    """The trivial predictors over the validation windows any cell answered.

    The mean, and for a remaining-life task the ceiling as well.
    """
    first = next(iter(cells.values()))
    targets = [prediction.target for prediction in first]
    baselines = [Baseline(name="mean predictor", rmse=pstdev(targets))]
    if isinstance(task.labels, RemainingLifeScheme):
        ceiling = task.labels.ceiling
        baselines.append(
            Baseline(
                name="ceiling predictor",
                rmse=(sum((ceiling - target) ** 2 for target in targets) / len(targets)) ** 0.5,
            )
        )
    return tuple(baselines)


def curve_of(shards: Sequence[Stored], task: KnownTask) -> Curve:
    """The curve the shards make, read by the registered rules.

    Raises:
        SystemExit: If the shards hold no whole cell, or do not make one grid.
    """
    require_one_configuration(shards, task)
    cells = predictions_of(shards)
    if not cells:
        raise SystemExit("the shards hold no prediction")
    return Curve(
        points=points_of(shards, cells, task),
        comparisons=comparisons_of(
            cells, task, PairedUnitBootstrap(resamples=RESAMPLES, seed=1), RULES
        ),
        baselines=baselines_of(cells, task),
    )


def write(curve: Curve, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        (CURVE, CurvePoint.COLUMNS, curve.points),
        (COMPARISONS, Comparison.COLUMNS, curve.comparisons),
        (BASELINES, Baseline.COLUMNS, curve.baselines),
    ):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(row.record() for row in rows)
    return directory


def read(directory: Path) -> Curve:
    """The curve stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CURVE).is_file():
        raise SystemExit(f"{directory} holds no {CURVE}; nothing to render")

    def rows(name: str) -> list[dict[str, str]]:
        with (directory / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    return Curve(
        points=tuple(CurvePoint.parse(row) for row in rows(CURVE)),
        comparisons=tuple(Comparison.parse(row) for row in rows(COMPARISONS)),
        baselines=tuple(Baseline.parse(row) for row in rows(BASELINES)),
    )


def sentence(curve: Curve) -> str:
    """The one-sentence conclusion the registered rules allow, and no more.

    The sentence is the Evaluation context's; what is composed here is the verdict it is read
    from, out of the stored rows, and the one thing a campaign never has to say: that the grid
    is not whole.
    """
    primary = next((row for row in curve.comparisons if row.primary), None)
    seeds = min((row.seeds for row in curve.comparisons), default=0)
    partial = ""
    if len(curve.comparisons) < COMPARED_CELLS or seeds < len(SEEDS):
        partial = (
            f"Incomplete grid — {len(curve.comparisons)} of {COMPARED_CELLS} cells compared, "
            f"{seeds} of {len(SEEDS)} seeds pooled — so not the registered reading. "
        )
    if primary is None:
        return f"{partial}The registered endpoint was not measured."
    verdict = CampaignVerdict(
        control=CandidateRef(str(CONTROL)),
        read_on=RunPurpose.TUNING,
        endpoint=_comparison_of(primary),
        secondary=tuple(_comparison_of(row) for row in curve.comparisons if not row.primary),
    )
    return f"{partial}{verdict.sentence()}"


def _comparison_of(row: Comparison) -> CandidateComparison:
    """The stored row as the context's comparison, its spread carried as it was stored."""
    return CandidateComparison(
        candidate=CandidateRef(row.mode),
        budget=LabelBudget.parse(row.budget),
        control_error=ErrorOverRepeats(
            pooled=row.control_rmse, spread=row.control_sd, repeats=row.seeds
        ),
        candidate_error=ErrorOverRepeats(
            pooled=row.candidate_rmse, spread=row.candidate_sd, repeats=row.seeds
        ),
        difference=PairedDifference(
            reduction=row.reduction,
            relative_reduction=row.relative_reduction,
            interval=BootstrapInterval(low=row.low, high=row.high, level=0.95),
            p_value=row.p_value,
        ),
        floor=PracticalFloor(value=row.floor),
        verdict=ComparisonVerdict(row.verdict),
    )


def render(curve: Curve, task: KnownTask) -> str:
    """The sections a note pastes, rendered from the three files."""
    budgets = sorted({row.budget for row in curve.points}, key=budget_rank)
    by_cell: dict[tuple[str, str], list[CurvePoint]] = {}
    for point in curve.points:
        by_cell.setdefault((point.mode, point.budget), []).append(point)

    def cell_rows(reading: str) -> list[list[str]]:
        rows = []
        for budget in budgets:
            engines = sorted(
                {p.engines for (_, b), ps in by_cell.items() if b == budget for p in ps}
            )
            row = [budget, f"{engines[0]}-{engines[-1]}" if len(engines) > 1 else str(engines[0])]
            for mode in MODES:
                points = by_cell.get((mode, budget), [])
                values = [
                    value
                    for value in (getattr(point, reading) for point in points)
                    if value is not None
                ]
                row.append(f"{mean(values):.2f} ± {_spread(values):.2f}" if values else "—")
            rows.append(row)
        return rows

    def comparison_rows() -> list[list[str]]:
        return [
            [
                row.mode,
                row.budget,
                f"{row.control_rmse:.2f}",
                f"{row.candidate_rmse:.2f}",
                f"{row.reduction:+.2f}",
                f"{row.relative_reduction:+.1%}",
                f"[{row.low:+.2f}, {row.high:+.2f}]",
                f"{row.p_value:.4f}",
                f"{row.floor:.2f}",
                "primary" if row.primary else ("yes" if row.rejected else "no"),
                row.verdict,
            ]
            for row in curve.comparisons
        ]

    units = task.units_called
    remaining_life_readings = (
        [
            "",
            "**Readings beside the endpoint**, mean ± SD over seeds:",
            "",
            "RMSE below the ceiling:",
            "",
            table(("budget", units, *MODES), cell_rows("rmse_below_ceiling")),
            "",
            "RMSE on the last window of each engine (on the validation side an engine's last "
            "window ends within four cycles of its failure, so this is the error at the end of "
            "life; the benchmark's protocol, and the only reading comparable with published "
            "numbers, is the same over the test side's trajectories cut short of failure):",
            "",
            table(("budget", units, *MODES), cell_rows("last_window_rmse")),
            "",
            "Alpha-lambda accuracy (share of answers within 20 % of the label):",
            "",
            table(("budget", units, *MODES), cell_rows("alpha_lambda_accuracy")),
            "",
            "Asymmetric score (mean per window, lower is better; its exponential tail is "
            "carried by a few engines):",
            "",
            table(("budget", units, *MODES), cell_rows("asymmetric_score")),
        ]
        if isinstance(task.labels, RemainingLifeScheme)
        else []
    )
    lines = [
        dated_heading(),
        "",
        f"Task `{task.name}`, {task.labels_text}, {task.strata} strata. Every number is "
        "validation, not test; the result is preliminary.",
        "",
        f"**Endpoint RMSE per cell, mean ± SD over seeds** (the {units} column is how many "
        f"{units} the budget's labels came from, over the seeds):",
        "",
        table(("budget", units, *MODES), cell_rows("rmse")),
        "",
        "Trivial predictors over the same windows: "
        + ", ".join(f"{b.name} {b.rmse:.2f}" for b in curve.baselines)
        + ".",
        "",
        "**Against the control arm**, pooled over the seeds both arms hold, the interval a "
        f"bootstrap over {units} (10,000 resamples); the endpoint stands alone and the rest are "
        f"the registered family of {COMPARED_CELLS - 1} under the Holm correction at 5 %, a "
        "cell that has not run counting as never rejected:",
        "",
        table(
            (
                "mode",
                "budget",
                "control",
                "candidate",
                "reduction",
                "relative",
                "95 % interval",
                "p",
                "floor",
                "rejected",
                "verdict",
            ),
            comparison_rows(),
        ),
        *remaining_life_readings,
        "",
        "Seconds per run:",
        "",
        table(("budget", units, *MODES), cell_rows("seconds")),
        "",
        f"**Conclusion.** {sentence(curve)}",
    ]
    return "\n".join(lines)


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shards", nargs="*", type=Path, help="directories the transfer report stored cells under"
    )
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored curve")
    parser.add_argument(
        "--task",
        choices=KnownTasks.names(),
        default=KnownTasks.default().name,
        help="the supervised task the shards were run on",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the curve is stored; data/report/curve/<first shard's name> unless given",
    )
    arguments = parser.parse_args(argv)
    if (arguments.report_only is None) == (not arguments.shards):
        parser.error("give the shards to read, or --report-only to render a stored curve")
    return arguments


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    task = KnownTasks.named(arguments.task)
    if arguments.report_only is not None:
        print(render(read(arguments.report_only), task))
        return
    shards = [Stored.existing(directory) for directory in arguments.shards]
    curve = curve_of(shards, task)
    out = arguments.out or REPO_ROOT / "data" / "report" / "curve" / shards[0].directory.name
    write(curve, out)
    print(render(curve, task))
    print(f"\nstored under {out}")


if __name__ == "__main__":
    main()
