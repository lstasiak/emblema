"""The readings a detection task reports beside its area, per campaign, candidate, budget and seed.

From an export of ``campaign_pairs_report.py``: the event-weighted area, the share of events found
with at most a stated share of normal windows flagged, and the benchmark's corrected event-wise
F-score at the detector's own alarm. The arithmetic is the harness's own ``AnomalyRanking``. The
export runs to tens of millions of rows and holds each cell's rows together, so one cell at a time
is held in memory. Writes a CSV; figures and tables read that.

    uv run scripts/detection_side_readings.py --predictions predictions.csv --out beside.csv
        [--labels campaigns.txt] [--share 0.01] [--beta 0.5]

``--labels`` names a file of lines ``label campaign-id ...``, for campaigns read by their label.
"""

import argparse
import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import groupby
from pathlib import Path

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction

COLUMNS = ("campaign", "candidate", "budget", "seed", "area", "event_recall", "event_fbeta")


def readings(
    rows: Iterable[Mapping[str, str]], share: float, beta: float, labels: Mapping[str, str]
) -> Iterator[tuple[str, str, str, int, float, float, float]]:
    """One reading per cell, cells taken as they lie in the export.

    Raises:
        SystemExit: If the rows of a cell do not lie together, which would read part of a cell.
    """
    seen: set[tuple[str, str, str, str]] = set()
    for (campaign, candidate, budget, seed), cell in groupby(
        rows, key=lambda row: (row["campaign"], row["candidate"], row["budget"], row["seed"])
    ):
        if (campaign, candidate, budget, seed) in seen:
            raise SystemExit(
                f"the rows of {campaign} {candidate} {budget} {seed} do not lie together"
            )
        seen.add((campaign, candidate, budget, seed))
        ranking = AnomalyRanking.of(
            WindowPrediction(
                window=TaskWindow(
                    unit=UnitKey(row["unit"]),
                    position=int(row["position"]),
                    ends_at=float(row["ends_at"]),
                ),
                target=float(row["target"]),
                predicted=float(row["predicted"]),
            )
            for row in cell
        )
        yield (
            labels.get(campaign, campaign),
            candidate,
            budget,
            int(seed),
            ranking.average_precision,
            ranking.event_recall_at_false_alarm_share(share),
            ranking.corrected_event_fbeta(beta),
        )


def read_labels(path: Path | None) -> dict[str, str]:
    """The label of each campaign, from lines ``label campaign-id ...``; comments are skipped."""
    if path is None:
        return {}
    labels: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and not line.startswith("#"):
            labels[fields[1]] = fields[0]
    return labels


def main(argv: Sequence[str] | None = None) -> None:
    """Read the export a cell at a time and write one row per cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--share", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    labels = read_labels(arguments.labels)
    count = 0
    with (
        arguments.predictions.open(newline="") as source,
        arguments.out.open("w", newline="") as target,
    ):
        writer = csv.writer(target)
        writer.writerow(COLUMNS)
        for row in readings(csv.DictReader(source), arguments.share, arguments.beta, labels):
            writer.writerow(row)
            count += 1
    print(f"{count} cells in {arguments.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
