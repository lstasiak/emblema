"""Put the published network's answers beside a campaign's exported cells, to be paired with them.

``strats_reference_run.py`` answers the scored stays once per seed. Written into the directory
``campaign_pairs_report.py`` exported a campaign to, as cells of a side of their own, they pair
with that campaign's candidates by the same paired bootstrap over stays the campaigns are read by.
Each answer takes the window its stay has in the export, so a ranking of the published network's
answers and a ranking of the campaign's hold the same windows. Rows of the side already there are
replaced, so writing again does not double them.

    uv run scripts/strats_reference_answers.py --into DIR --side NAME CANDIDATE
        --purpose selection --runs RUN_DIR [RUN_DIR ...]
    uv run scripts/campaign_pairs_report.py --out DIR
        --pair CAMPAIGN CANDIDATE all NAME CANDIDATE all
"""

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from scripts.campaign_pairs_report import (
    CELLS,
    PREDICTIONS,
    CellRow,
    PredictionRow,
    read_cells,
    read_predictions,
    write_cells,
    write_predictions,
)

# Every stay is one window learnt from in full, as the campaigns' budget of every stay is named.
BUDGET = "all"
TIER = "M"


class UnpairableAnswersError(ValueError):
    """Raised where a run's answers cannot stand beside the export."""


def runs_answers(run: Path) -> tuple[list[tuple[int, str, float, float]], float]:
    """A run's answers, as (seed, stay, target, answer), and the seconds it trained for."""
    with (run / "predictions.csv").open(newline="") as file:
        rows = [
            (int(line["seed"]), line["unit"], float(line["target"]), float(line["predicted"]))
            for line in csv.DictReader(file)
        ]
    seconds = float(json.loads((run / "run.json").read_text())["seconds"])
    return rows, seconds


def windows_of(predictions: Iterable[PredictionRow]) -> dict[str, tuple[int, float, float]]:
    """Each stay's window in the export, with its target: position, end and label.

    Raises:
        UnpairableAnswersError: If the export gives a stay two windows or two targets.
    """
    windows: dict[str, tuple[int, float, float]] = {}
    for row in predictions:
        window = (row.position, row.ends_at, row.target)
        if windows.setdefault(row.unit, window) != window:
            raise UnpairableAnswersError(f"the export holds {row.unit} in more than one window")
    return windows


def side_rows(
    answers: Iterable[tuple[int, str, float, float]],
    seconds: Mapping[int, float],
    windows: Mapping[str, tuple[int, float, float]],
    *,
    name: str,
    candidate: str,
    purpose: str,
) -> tuple[list[CellRow], list[PredictionRow]]:
    """The answers as the export's cells and predictions, under the side ``name`` and ``candidate``.

    Raises:
        UnpairableAnswersError: If a stay has no window in the export, or its target differs
            from the export's.
    """
    cells, predictions = [], []
    for seed, stay, target, answer in answers:
        if stay not in windows:
            raise UnpairableAnswersError(f"{stay} has no window in the export")
        position, ends_at, expected = windows[stay]
        if target != expected:
            raise UnpairableAnswersError(
                f"{stay} is labelled {target} here and {expected} in the export"
            )
        cells.append(
            CellRow(
                campaign=name,
                purpose=purpose,
                tier=TIER,
                measure=ErrorMeasure.AUROC_SHORTFALL.value,
                candidate=candidate,
                budget=BUDGET,
                seed=seed,
                unit=stay,
                squared_error=(answer - target) ** 2,
                windows=1,
                seconds=seconds[seed],
            )
        )
        predictions.append(
            PredictionRow(
                campaign=name,
                candidate=candidate,
                budget=BUDGET,
                seed=seed,
                unit=stay,
                position=position,
                ends_at=ends_at,
                target=target,
                predicted=answer,
            )
        )
    return cells, predictions


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--into", type=Path, required=True, help="a campaign export directory")
    parser.add_argument("--side", nargs=2, required=True, metavar=("NAME", "CANDIDATE"))
    parser.add_argument(
        "--purpose", required=True, help="the purpose of the campaign it pairs with"
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    name, candidate = arguments.side
    cells = read_cells(arguments.into / CELLS)
    predictions = read_predictions(arguments.into / PREDICTIONS)
    answers: list[tuple[int, str, float, float]] = []
    seconds: dict[int, float] = {}
    for run in arguments.runs:
        rows, spent = runs_answers(run)
        repeated = sorted({seed for seed, _, _, _ in rows} & seconds.keys())
        if repeated:
            sys.exit(f"{run} answers again for seeds {repeated}; one run per seed")
        answers.extend(rows)
        seconds.update((seed, spent) for seed, _, _, _ in rows)
    kept = [row for row in predictions if (row.campaign, row.candidate) != (name, candidate)]
    try:
        added_cells, added_predictions = side_rows(
            answers,
            seconds,
            windows_of(kept),
            name=name,
            candidate=candidate,
            purpose=arguments.purpose,
        )
    except UnpairableAnswersError as error:
        sys.exit(str(error))
    write_cells(
        arguments.into / CELLS,
        [row for row in cells if (row.campaign, row.candidate) != (name, candidate)] + added_cells,
    )
    write_predictions(arguments.into / PREDICTIONS, kept + added_predictions)
    print(f"{len(added_predictions)} answers of {name} {candidate} over seeds {sorted(seconds)}")


if __name__ == "__main__":
    main()
