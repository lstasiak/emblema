"""Read a finished comparison out of the registry: its cells, its verdict, the note's tables.

The campaign judges its own grid by the rules its design registered; what a note needs beside
the verdict is the grid as numbers a reader can check — a row per cell and repeat with its
error, a row per comparison with the interval and the floor it was held to — and the tables
those rows make. Nothing here computes what the verdict rests on: the arithmetic is the
Evaluation context's, and this writes it down.

    uv run scripts/campaign_report.py --campaign ID --out DIR [--everything N]

writes ``cells.csv``, ``comparisons.csv`` and ``verdict.md`` under DIR, each in the terms of the
campaign's measure: an error in the task's unit, or the area under the ROC curve with the Brier
score beside it. A campaign whose grid
runs at every labelled window there is needs ``--everything``, the count that budget resolved
to, which the registry does not hold and a figure needs for its axis.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import NamedTuple

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.contracts.identifiers import CampaignId
from emblema.evaluation.domain.campaign.campaign_reading import CampaignReading
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.exceptions import EvaluationError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.ports.evaluation_campaign_repository import (
    EvaluationCampaignRepository,
)
from scripts.reporting import table

CELLS, COMPARISONS, VERDICT = "cells.csv", "comparisons.csv", "verdict.md"
CELL_COLUMNS = ("candidate", "kind", "budget", "windows", "seed", "units", "rmse", "seconds")
CELL_COLUMNS_BY_AREA = (
    "candidate",
    "kind",
    "budget",
    "windows",
    "seed",
    "units",
    "auroc",
    "brier",
    "seconds",
)
COMPARISON_COLUMNS = (
    "candidate",
    "budget",
    "windows",
    "repeats",
    "control_rmse",
    "control_sd",
    "candidate_rmse",
    "candidate_sd",
    "reduction",
    "relative_reduction",
    "low",
    "high",
    "p_value",
    "floor",
    "primary",
    "verdict",
)
# The same figures read by area: the scores are areas, and the reduction of one minus the area is
# the gain in area, so the interval and the floor are the gain's.
COMPARISON_COLUMNS_BY_AREA = (
    "candidate",
    "budget",
    "windows",
    "repeats",
    "control_auroc",
    "control_sd",
    "candidate_auroc",
    "candidate_sd",
    "gain",
    "share_of_shortfall",
    "low",
    "high",
    "p_value",
    "floor",
    "primary",
    "verdict",
)


class CellRow(NamedTuple):
    """One cell under one seed: what it scored, in the measure's natural terms, and what it cost.

    The score is the error for a measure that is one, and the area for a ranking; the Brier score
    is there only for the latter.
    """

    candidate: str
    kind: str
    budget: str
    windows: int
    seed: int
    units: int
    score: float
    brier: float | None
    seconds: float

    def written(self) -> tuple[object, ...]:
        """The row as its file holds it, without the column its measure does not have."""
        head = (self.candidate, self.kind, self.budget, self.windows, self.seed, self.units)
        if self.brier is None:
            return (*head, self.score, self.seconds)
        return (*head, self.score, self.brier, self.seconds)


class ComparisonRow(NamedTuple):
    """One candidate against the control at one budget, as the campaign judged it.

    The relative reduction is empty where the control made no error, so no share exists.
    """

    candidate: str
    budget: str
    windows: int
    repeats: int
    control_score: float
    control_sd: float
    candidate_score: float
    candidate_sd: float
    reduction: float
    relative_reduction: float | None
    low: float
    high: float
    p_value: float
    floor: float
    primary: str
    verdict: str


def windows_of(budget: LabelBudget, everything: int | None) -> int:
    """The count a budget resolved to.

    Raises:
        SystemExit: If the budget is every window and the count was not given.
    """
    if budget.windows is not None:
        return budget.windows
    if everything is None:
        raise SystemExit(
            "the campaign runs at every labelled window; give --everything, the count that "
            "budget resolved to"
        )
    return everything


def by_area(reading: CampaignReading) -> bool:
    """Whether the campaign reads its candidates by how they rank, not by an error."""
    return reading.campaign.design.measure is ErrorMeasure.AUROC_SHORTFALL


def natural(reading: CampaignReading, error: float) -> float:
    """An error of the campaign's measure as a reader states it: an area, or the error itself."""
    return 1.0 - error if by_area(reading) else error


def export_cells(reading: CampaignReading, everything: int | None) -> list[CellRow]:
    """Every recorded cell, in the grid's order: candidate, then budget, then seed."""
    design = reading.campaign.design
    recorded = {result.cell: result for result in reading.results}
    return [
        CellRow(
            candidate=str(cell.candidate),
            kind=str(design.get_candidate(cell.candidate).kind),
            budget=cell.budget.text(),
            windows=windows_of(cell.budget, everything),
            seed=cell.seed,
            units=len(recorded[cell].errors),
            score=natural(reading, recorded[cell].error_under(design.measure)),
            brier=recorded[cell].mean_squared_error if by_area(reading) else None,
            seconds=recorded[cell].seconds,
        )
        for cell in design.cells()
        if cell in recorded
    ]


def export_comparisons(
    verdict: CampaignVerdict, reading: CampaignReading, everything: int | None
) -> list[ComparisonRow]:
    """Every comparison the campaign made, the endpoint first."""
    return [
        ComparisonRow(
            candidate=str(compared.candidate),
            budget=compared.budget.text(),
            windows=windows_of(compared.budget, everything),
            repeats=compared.control_error.repeats,
            control_score=natural(reading, compared.control_error.pooled),
            control_sd=compared.control_error.spread,
            candidate_score=natural(reading, compared.candidate_error.pooled),
            candidate_sd=compared.candidate_error.spread,
            reduction=compared.difference.reduction,
            relative_reduction=compared.difference.relative_reduction,
            low=compared.difference.interval.low,
            high=compared.difference.interval.high,
            p_value=compared.difference.p_value,
            floor=compared.floor.value,
            primary="yes" if compared is verdict.endpoint else "no",
            verdict=str(compared.verdict),
        )
        for compared in verdict.comparisons()
    ]


def share_of(compared: ComparisonRow) -> str:
    """The relative reduction as a table shows it; a control without error has no share."""
    if compared.relative_reduction is None:
        return "n/a (control without error)"
    return f"{compared.relative_reduction:+.1%}"


def render(
    reading: CampaignReading,
    verdict: CampaignVerdict,
    cells: Sequence[CellRow],
    comparisons: Sequence[ComparisonRow],
) -> str:
    """The sentence and one table per budget: every candidate's error, and its standing."""
    judged = {(row.candidate, row.budget): row for row in comparisons}
    design = reading.campaign.design
    area = by_area(reading)
    score = "AUROC" if area else "RMSE"
    parts = [verdict.sentence(), ""]
    for budget in design.budgets:
        rows = []
        for candidate in design.candidates:
            scored = [
                c.score
                for c in cells
                if c.candidate == str(candidate.ref) and c.budget == budget.text()
            ]
            if not scored:
                continue
            spread = stdev(scored) if len(scored) > 1 else 0.0
            error = f"{mean(scored):.2f} ± {spread:.2f}"
            compared = judged.get((str(candidate.ref), budget.text()))
            if compared is None:
                rows.append((str(candidate.ref), error, "control", "", ""))
                continue
            rows.append(
                (
                    str(candidate.ref),
                    error,
                    f"{compared.reduction:+.3f}" if area else share_of(compared),
                    f"[{compared.low:+.2f}, {compared.high:+.2f}]",
                    compared.verdict + (" (endpoint)" if compared.primary == "yes" else ""),
                )
            )
        parts.append(f"**{budget.text()} labelled windows**, {score} mean ± SD over the repeats.")
        parts.append("")
        change = "gain in area vs control" if area else "reduction vs control"
        parts.append(table(("candidate", score, change, "95 % interval", "verdict"), rows))
        parts.append("")
    return "\n".join(parts)


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[tuple[object, ...]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([repr(v) if isinstance(v, float) else v for v in row])


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, metavar="ID", help="a finished comparison")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument(
        "--everything",
        type=int,
        help="how many labelled windows the budget of every window resolved to",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, *, registry: EvaluationCampaignRepository | None = None
) -> None:
    """Read the campaign, write its three files, print the sentence.

    Raises:
        SystemExit: If the campaign is not a finished comparison, or the count of every
            labelled window is needed and was not given.
    """
    arguments = parse_arguments(argv)
    campaigns = _registry() if registry is None else registry
    reading = campaigns.read(CampaignId.parse(arguments.campaign))
    try:
        verdict = reading.verdict()
    except EvaluationError as refusal:
        raise SystemExit(str(refusal)) from refusal
    cells = export_cells(reading, arguments.everything)
    comparisons = export_comparisons(verdict, reading, arguments.everything)
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    area = by_area(reading)
    write_rows(
        out / CELLS, CELL_COLUMNS_BY_AREA if area else CELL_COLUMNS, [c.written() for c in cells]
    )
    write_rows(
        out / COMPARISONS,
        COMPARISON_COLUMNS_BY_AREA if area else COMPARISON_COLUMNS,
        [tuple(row) for row in comparisons],
    )
    rendered = render(reading, verdict, cells, comparisons)
    (out / VERDICT).write_text(rendered + "\n")
    print(rendered)


def _registry() -> EvaluationCampaignRepository:  # pragma: no cover - environment
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_engine
    from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
        SqlAlchemyEvaluationCampaignRepository,
    )

    return SqlAlchemyEvaluationCampaignRepository(configured_engine(Settings()))


if __name__ == "__main__":
    main()
