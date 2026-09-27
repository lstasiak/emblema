"""Read a finished comparison out of the registry: its cells, its verdict, the note's tables.

The campaign judges its own grid by the rules its design registered; what a note needs beside
the verdict is the grid as numbers a reader can check — a row per cell and repeat with its
error, a row per comparison with the interval and the floor it was held to — and the tables
those rows make. Nothing here computes what the verdict rests on: the arithmetic is the
Evaluation context's, and this writes it down.

    uv run scripts/campaign_report.py --campaign ID --out DIR [--everything N]

writes ``cells.csv``, ``comparisons.csv`` and ``verdict.md`` under DIR. A campaign whose grid
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
from emblema.evaluation.domain.campaign.campaign_verdict import CampaignVerdict
from emblema.evaluation.domain.campaign.evaluation_campaign import EvaluationCampaign
from emblema.evaluation.domain.exceptions import EvaluationError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.ports.evaluation_campaign_repository import (
    EvaluationCampaignRepository,
)
from scripts.reporting import table

CELLS, COMPARISONS, VERDICT = "cells.csv", "comparisons.csv", "verdict.md"
CELL_COLUMNS = ("candidate", "kind", "budget", "windows", "seed", "units", "rmse", "seconds")
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


class CellRow(NamedTuple):
    """One cell under one seed: what it scored and what it cost."""

    candidate: str
    kind: str
    budget: str
    windows: int
    seed: int
    units: int
    rmse: float
    seconds: float


class ComparisonRow(NamedTuple):
    """One candidate against the control at one budget, as the campaign judged it."""

    candidate: str
    budget: str
    windows: int
    repeats: int
    control_rmse: float
    control_sd: float
    candidate_rmse: float
    candidate_sd: float
    reduction: float
    relative_reduction: float
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


def export_cells(campaign: EvaluationCampaign, everything: int | None) -> list[CellRow]:
    """Every recorded cell, in the grid's order: candidate, then budget, then seed."""
    recorded = {result.cell: result for result in campaign.results}
    return [
        CellRow(
            candidate=str(cell.candidate),
            kind=str(campaign.design.get_candidate(cell.candidate).kind),
            budget=cell.budget.text(),
            windows=windows_of(cell.budget, everything),
            seed=cell.seed,
            units=len(recorded[cell].errors),
            rmse=recorded[cell].rmse,
            seconds=recorded[cell].seconds,
        )
        for cell in campaign.design.cells()
        if cell in recorded
    ]


def export_comparisons(
    verdict: CampaignVerdict, campaign: EvaluationCampaign, everything: int | None
) -> list[ComparisonRow]:
    """Every comparison the campaign made, the endpoint first."""
    return [
        ComparisonRow(
            candidate=str(compared.candidate),
            budget=compared.budget.text(),
            windows=windows_of(compared.budget, everything),
            repeats=compared.control_error.repeats,
            control_rmse=compared.control_error.pooled,
            control_sd=compared.control_error.spread,
            candidate_rmse=compared.candidate_error.pooled,
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


def render(
    campaign: EvaluationCampaign,
    verdict: CampaignVerdict,
    cells: Sequence[CellRow],
    comparisons: Sequence[ComparisonRow],
) -> str:
    """The sentence and one table per budget: every candidate's error, and its standing."""
    judged = {(row.candidate, row.budget): row for row in comparisons}
    parts = [verdict.sentence(), ""]
    for budget in campaign.design.budgets:
        rows = []
        for candidate in campaign.design.candidates:
            scored = [
                c.rmse
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
                    f"{compared.relative_reduction:+.1%}",
                    f"[{compared.low:+.2f}, {compared.high:+.2f}]",
                    compared.verdict + (" (endpoint)" if compared.primary == "yes" else ""),
                )
            )
        parts.append(f"**{budget.text()} labelled windows**, RMSE mean ± SD over the repeats.")
        parts.append("")
        parts.append(
            table(("candidate", "RMSE", "reduction vs control", "95 % interval", "verdict"), rows)
        )
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
    campaign = campaigns.get(CampaignId.parse(arguments.campaign))
    try:
        verdict = campaign.verdict()
    except EvaluationError as refusal:
        raise SystemExit(str(refusal)) from refusal
    cells = export_cells(campaign, arguments.everything)
    comparisons = export_comparisons(verdict, campaign, arguments.everything)
    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / CELLS, CELL_COLUMNS, [tuple(row) for row in cells])
    write_rows(out / COMPARISONS, COMPARISON_COLUMNS, [tuple(row) for row in comparisons])
    rendered = render(campaign, verdict, cells, comparisons)
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
