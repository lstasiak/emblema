"""Set a cell of one campaign beside a cell of another, paired on the units both scored.

A campaign judges its own grid, and every network in it spends one compute budget. What it
cannot ask is whether the same arm wants twice the steps: that is a second campaign under a
doubled floor, and the two are read side by side. This script reads the cells of the named
campaigns out of the registry into one CSV — a row per cell and unit, with the purpose and the
tier the campaign was signed with — and pairs any two sides, a (campaign, candidate, budget)
each, with the paired bootstrap the harness judges by. The arithmetic is the Evaluation
context's; what is here is composition and the shape of a note.

How a pair is made depends on what the campaigns were for. A selection divides the tuning side
afresh under every seed, so its repeats hold out different units and the pair is made over
(repeat, unit); a comparison scores every repeat on the same validation units, so its repeats
are pooled per unit first, as the registered rules do. Both sides must have been run for the
same purpose, on the same seeds and over the same units.

A campaign read by the area under the ROC curve is paired as its verdict pairs it: from every
window's answer, each repeat ranked on its own and the repeats of a side pooled by the mean of
their areas, the units resampled in two strata, the floor stated in area. Only comparisons pair
that way, since a selection's repeats score other units and two models' answers are never ranked
together.

    uv run scripts/campaign_pairs_report.py --campaign ID [--campaign ID ...] --out DIR
    uv run scripts/campaign_pairs_report.py --out DIR
        --pair CONTROL_ID CONTROL_CANDIDATE BUDGET CANDIDATE_ID CANDIDATE BUDGET [--pair ...]

The first form writes ``cells.csv``, and ``predictions.csv`` for campaigns read by area; the
second reads them, writes ``comparisons.csv`` and prints the table a note pastes. Both may be
given at once.
"""

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import NamedTuple

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.paired_difference import PairedDifference
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings
from emblema.evaluation.domain.statistics.paired_units import PairedUnits
from emblema.evaluation.domain.statistics.practical_floor import PracticalFloor
from emblema.evaluation.domain.statistics.threshold_kind import ThresholdKind
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from scripts.reporting import table

CELLS, PREDICTIONS, COMPARISONS = "cells.csv", "predictions.csv", "comparisons.csv"
CELL_COLUMNS = (
    "campaign",
    "purpose",
    "tier",
    "measure",
    "candidate",
    "budget",
    "seed",
    "unit",
    "squared_error",
    "windows",
    "seconds",
)
COMPARISON_COLUMNS = (
    "control_campaign",
    "control",
    "candidate_campaign",
    "candidate",
    "budget",
    "pairing",
    "repeats",
    "units",
    "error_control",
    "spread_control",
    "error_candidate",
    "spread_candidate",
    "reduction",
    "relative_reduction",
    "low",
    "high",
    "p_value",
    "floor",
    "measure",
)
PREDICTION_COLUMNS = (
    "campaign",
    "candidate",
    "budget",
    "seed",
    "unit",
    "position",
    "ends_at",
    "target",
    "predicted",
)


class CellRow(NamedTuple):
    """One unit of one cell, as the CSV holds it."""

    campaign: str
    purpose: str
    tier: str
    measure: str
    candidate: str
    budget: str
    seed: int
    unit: str
    squared_error: float
    windows: int
    seconds: float


class PredictionRow(NamedTuple):
    """One window's answer in one cell, as the CSV holds it, for a campaign read by area."""

    campaign: str
    candidate: str
    budget: str
    seed: int
    unit: str
    position: int
    ends_at: float
    target: float
    predicted: float

    def prediction(self) -> WindowPrediction:
        return WindowPrediction(
            window=TaskWindow(
                unit=UnitKey(self.unit), position=self.position, ends_at=self.ends_at
            ),
            target=self.target,
            predicted=self.predicted,
        )


class Side(NamedTuple):
    """One side of a pair: a candidate at a budget of one campaign."""

    campaign: str
    candidate: str
    budget: str

    def __str__(self) -> str:
        return f"{self.candidate} at {self.budget} of {self.campaign[:8]}"


class Answers:
    """Every exported answer, looked up by the repeat of a side it belongs to.

    Indexed once, because a full grid read by area holds millions of answers and a pair asks
    for one repeat of each side in turn.
    """

    def __init__(self, rows: Iterable[PredictionRow]) -> None:
        self._by_repeat: dict[tuple[str, str, str, int], list[WindowPrediction]] = {}
        for row in rows:
            key = (row.campaign, row.candidate, row.budget, row.seed)
            self._by_repeat.setdefault(key, []).append(row.prediction())

    def ranking(self, side: Side, seed: int) -> WindowRanking:
        """One repeat of a side, its windows ranked by their answers.

        Raises:
            ValueError: If no answer of that repeat was exported.
        """
        answers = self._by_repeat.get((side.campaign, side.candidate, side.budget, seed))
        if not answers:
            raise ValueError(
                f"no answer of {side} under seed {seed} among the exported predictions"
            )
        return WindowRanking.of(answers)


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """One pair read: what each side scored, and the paired difference between them."""

    control: Side
    candidate: Side
    measure: ErrorMeasure
    pairing: str
    repeats: int
    units: int
    control_error: ErrorOverRepeats
    candidate_error: ErrorOverRepeats
    difference: PairedDifference
    floor: PracticalFloor

    def row(self) -> tuple[str, ...]:
        share = self.difference.relative_reduction
        return (
            self.control.campaign,
            self.control.candidate,
            self.candidate.campaign,
            self.candidate.candidate,
            self.control.budget,
            self.pairing,
            str(self.repeats),
            str(self.units),
            repr(self.control_error.pooled),
            repr(self.control_error.spread),
            repr(self.candidate_error.pooled),
            repr(self.candidate_error.spread),
            repr(self.difference.reduction),
            "" if share is None else repr(share),
            repr(self.difference.interval.low),
            repr(self.difference.interval.high),
            repr(self.difference.p_value),
            repr(self.floor.value),
            self.measure.value,
        )


def export(readings: Iterable[CampaignReading]) -> list[CellRow]:
    """Every unit of every recorded cell of ``readings``, in the order they were recorded."""
    return [
        CellRow(
            campaign=str(reading.campaign.campaign_id),
            purpose=reading.campaign.purpose.value,
            tier=str(reading.campaign.tier),
            measure=reading.campaign.design.measure.value,
            candidate=str(result.cell.candidate),
            budget=result.cell.budget.text(),
            seed=result.cell.seed,
            unit=str(error.unit),
            squared_error=error.squared_error,
            windows=error.windows,
            seconds=result.seconds,
        )
        for reading in readings
        for result in reading.results
        for error in result.errors
    ]


def export_predictions(readings: Iterable[CampaignReading]) -> list[PredictionRow]:
    """Every window's answer in every recorded cell of the campaigns read by area."""
    return [
        PredictionRow(
            campaign=str(reading.campaign.campaign_id),
            candidate=str(result.cell.candidate),
            budget=result.cell.budget.text(),
            seed=result.cell.seed,
            unit=str(prediction.window.unit),
            position=prediction.window.position,
            ends_at=prediction.window.ends_at,
            target=prediction.target,
            predicted=prediction.predicted,
        )
        for reading in readings
        if reading.campaign.design.measure is ErrorMeasure.AUROC_SHORTFALL
        for result in reading.results
        for prediction in result.predictions
    ]


def write_cells(path: Path, rows: Sequence[CellRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CELL_COLUMNS)
        for row in rows:
            writer.writerow(
                (*row[:8], repr(row.squared_error), row.windows, repr(row.seconds)),
            )


def read_cells(path: Path) -> list[CellRow]:
    with path.open(newline="") as handle:
        return [
            CellRow(
                campaign=line["campaign"],
                purpose=line["purpose"],
                tier=line["tier"],
                # A file written before campaigns were read by area holds errors only.
                measure=line.get("measure", ErrorMeasure.RMSE.value),
                candidate=line["candidate"],
                budget=line["budget"],
                seed=int(line["seed"]),
                unit=line["unit"],
                squared_error=float(line["squared_error"]),
                windows=int(line["windows"]),
                seconds=float(line["seconds"]),
            )
            for line in csv.DictReader(handle)
        ]


def write_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    *row[:4],
                    row.unit,
                    row.position,
                    repr(row.ends_at),
                    repr(row.target),
                    repr(row.predicted),
                )
            )


def read_predictions(path: Path) -> list[PredictionRow]:
    """The answers stored at ``path``; none where no campaign read by area was exported."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [
            PredictionRow(
                campaign=line["campaign"],
                candidate=line["candidate"],
                budget=line["budget"],
                seed=int(line["seed"]),
                unit=line["unit"],
                position=int(line["position"]),
                ends_at=float(line["ends_at"]),
                target=float(line["target"]),
                predicted=float(line["predicted"]),
            )
            for line in csv.DictReader(handle)
        ]


def pair(
    rows: Sequence[CellRow],
    control: Side,
    candidate: Side,
    *,
    bootstrap: PairedUnitBootstrap,
    floor_share: float,
    floor_area: float,
    answers: Answers | None = None,
) -> Comparison:
    """``candidate`` against ``control``, paired as the campaigns' purpose and measure say.

    Raises:
        ValueError: If a side has no cell, the sides were run for different purposes, read by
            different measures or run on different seeds, a selection is to be paired by area,
            or the units of a repeat do not pair.
    """
    control_cells = _cells_of(rows, control)
    candidate_cells = _cells_of(rows, candidate)
    purposes = {row.purpose for row in control_cells} | {row.purpose for row in candidate_cells}
    if len(purposes) != 1:
        raise ValueError(f"the sides were run for different purposes: {sorted(purposes)}")
    measures = {row.measure for row in control_cells} | {row.measure for row in candidate_cells}
    if len(measures) != 1:
        raise ValueError(f"the sides were read by different measures: {sorted(measures)}")
    measure = ErrorMeasure(measures.pop())
    seeds = sorted({row.seed for row in control_cells})
    if seeds != sorted({row.seed for row in candidate_cells}):
        raise ValueError(f"{control} and {candidate} were not run on the same seeds")
    paired: PairedUnits
    match measure:
        case ErrorMeasure.RMSE:
            control_repeats = [_errors_of(control_cells, seed) for seed in seeds]
            candidate_repeats = [_errors_of(candidate_cells, seed) for seed in seeds]
            if purposes == {RunPurpose.SELECTION.value}:
                pairing = "repeat-unit"
                paired = PairedUnitErrors(
                    control=_stamped(control_repeats, seeds),
                    candidate=_stamped(candidate_repeats, seeds),
                )
            else:
                pairing = "unit-pooled"
                paired = PairedUnitErrors.pooled(control_repeats, candidate_repeats)
            control_spread = [_rmse(r) for r in control_repeats]
            candidate_spread = [_rmse(r) for r in candidate_repeats]
            floor_part, threshold = floor_share, ThresholdKind.RELATIVE
        case ErrorMeasure.AUROC_SHORTFALL:
            if purposes == {RunPurpose.SELECTION.value}:
                raise ValueError(
                    "a selection read by area scores other units under every seed, and two "
                    "models' answers are never ranked together; only comparisons pair by area"
                )
            pairing = "area-pooled"
            found = Answers(()) if answers is None else answers
            rankings = PairedUnitRankings(
                control=tuple(found.ranking(control, seed) for seed in seeds),
                candidate=tuple(found.ranking(candidate, seed) for seed in seeds),
            )
            paired = rankings
            control_spread = list(rankings.per_repeat_control())
            candidate_spread = list(rankings.per_repeat_candidate())
            floor_part, threshold = floor_area, ThresholdKind.ABSOLUTE
    control_error = ErrorOverRepeats.of(paired.error_control, control_spread)
    candidate_error = ErrorOverRepeats.of(paired.error_candidate, candidate_spread)
    return Comparison(
        control=control,
        candidate=candidate,
        measure=measure,
        pairing=pairing,
        repeats=len(seeds),
        units=len(paired.units),
        control_error=control_error,
        candidate_error=candidate_error,
        difference=bootstrap.compare(paired),
        floor=PracticalFloor.of(control_error, part=floor_part, threshold=threshold),
    )


def _cells_of(rows: Sequence[CellRow], side: Side) -> list[CellRow]:
    found = [
        row
        for row in rows
        if (row.campaign, row.candidate, row.budget) == (side.campaign, side.candidate, side.budget)
    ]
    if not found:
        raise ValueError(f"no cell of {side} among the exported cells")
    return found


def _errors_of(rows: Sequence[CellRow], seed: int) -> tuple[UnitError, ...]:
    return tuple(
        UnitError(unit=UnitKey(row.unit), squared_error=row.squared_error, windows=row.windows)
        for row in sorted((row for row in rows if row.seed == seed), key=lambda row: row.unit)
    )


def _stamped(repeats: Sequence[Sequence[UnitError]], seeds: Sequence[int]) -> tuple[UnitError, ...]:
    """Every (repeat, unit) as a unit of its own, so repeats over different units still pair."""
    return tuple(
        UnitError(
            unit=UnitKey(f"{seed}/{error.unit}"),
            squared_error=error.squared_error,
            windows=error.windows,
        )
        for seed, errors in zip(seeds, repeats, strict=True)
        for error in errors
    )


def _rmse(errors: Sequence[UnitError]) -> float:
    return sqrt(sum(e.squared_error for e in errors) / sum(e.windows for e in errors))


def write_comparisons(path: Path, comparisons: Sequence[Comparison]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COMPARISON_COLUMNS)
        for comparison in comparisons:
            writer.writerow(comparison.row())


def share_of(difference: PairedDifference) -> str:
    """The relative reduction as a table shows it; a control without error has no share."""
    if difference.relative_reduction is None:
        return "no share: control without error"
    return f"{difference.relative_reduction:+.1%}"


def render(comparisons: Sequence[Comparison]) -> str:
    """The table a note pastes: each side's error over its repeats, the difference, the floor.

    Raises:
        ValueError: If the comparisons are read by different measures, which one table's
            columns cannot name.
    """
    measures = {comparison.measure for comparison in comparisons}
    if len(measures) > 1:
        raise ValueError(f"one table holds one measure, and these are read by {len(measures)}")
    error = "1 - AUROC" if measures == {ErrorMeasure.AUROC_SHORTFALL} else "RMSE"
    rows = [
        (
            str(comparison.control),
            str(comparison.control_error),
            str(comparison.candidate),
            str(comparison.candidate_error),
            f"{comparison.difference.reduction:+.3f} ({share_of(comparison.difference)})",
            str(comparison.difference.interval),
            f"{comparison.difference.p_value:.3f}",
            f"{comparison.floor.value:.3f}",
            f"{comparison.pairing}, {comparison.repeats} repeats, {comparison.units} pairs",
        )
        for comparison in comparisons
    ]
    return table(
        (
            "control",
            error,
            "candidate",
            error,
            "reduction",
            "95 % interval",
            "p",
            "floor",
            "paired",
        ),
        rows,
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        metavar="ID",
        help="a campaign to export the cells of; repeatable",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        nargs=6,
        metavar=("CONTROL_ID", "CONTROL", "BUDGET", "CANDIDATE_ID", "CANDIDATE", "BUDGET"),
        help="a candidate at a budget of one campaign against one of another; repeatable",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1, help="seed of the bootstrap's draws")
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument(
        "--floor-share", type=float, default=0.02, help="share of the control's error"
    )
    parser.add_argument(
        "--floor-area",
        type=float,
        default=0.01,
        help="fixed part of the floor in area, for campaigns read by area",
    )
    arguments = parser.parse_args(argv)
    if not arguments.campaign and not arguments.pair:
        parser.error("give a campaign to export, a pair to read, or both")
    return arguments


def main(
    argv: Sequence[str] | None = None, *, registry: EvaluationCampaignRepository | None = None
) -> None:
    """Export what is asked for, then read the pairs off the exported cells.

    Args:
        argv: The command line.
        registry: Where the campaigns are read from; the configured database unless given.
    """
    arguments = parse_arguments(argv)
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    if arguments.campaign:
        campaigns = _registry() if registry is None else registry
        readings = [campaigns.read(CampaignId.parse(each)) for each in arguments.campaign]
        rows = export(readings)
        write_cells(out / CELLS, rows)
        print(f"{len(rows)} unit errors of {len(arguments.campaign)} campaigns in {out / CELLS}")
        # Written even when empty, so no answers of an earlier export stay beside these cells.
        answers = export_predictions(readings)
        write_predictions(out / PREDICTIONS, answers)
        print(f"{len(answers)} answers of the campaigns read by area in {out / PREDICTIONS}")
    if arguments.pair:
        rows = read_cells(out / CELLS)
        found = Answers(read_predictions(out / PREDICTIONS))
        bootstrap = PairedUnitBootstrap(
            resamples=arguments.resamples, seed=arguments.seed, level=arguments.level
        )
        comparisons = [
            pair(
                rows,
                Side(*each[:3]),
                Side(*each[3:]),
                bootstrap=bootstrap,
                floor_share=arguments.floor_share,
                floor_area=arguments.floor_area,
                answers=found,
            )
            for each in arguments.pair
        ]
        write_comparisons(out / COMPARISONS, comparisons)
        print(render(comparisons))


def _registry() -> EvaluationCampaignRepository:  # pragma: no cover - environment
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_engine
    from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
        SqlAlchemyEvaluationCampaignRepository,
    )

    return SqlAlchemyEvaluationCampaignRepository(configured_engine(Settings()))


if __name__ == "__main__":
    main()
