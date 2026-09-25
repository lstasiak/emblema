"""How well the registered interval keeps its word, measured on data with a known answer.

The paired bootstrap over units is the procedure every verdict rests on, and a percentile interval
over a few dozen units is known to run short of its level. This measures by how much, on the
project's own known-answer generator: how often a 95 % interval covers a true reduction, and how
often it excludes a true difference of zero, over many synthetic datasets at several unit counts.
The numbers are written to CSV first and rendered as a table from that file, so the table can be
reshaped without drawing the datasets again.

    uv run scripts/bootstrap_calibration_report.py --out DIR
    uv run scripts/bootstrap_calibration_report.py --report-only DIR
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.synthetic.known_answer import KnownAnswer
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.reporting import dated_heading, table

CALIBRATION = "calibration.csv"
CONTROL_RMSE = 30.0
CANDIDATE_RMSE = 25.0
UNIT_COUNTS = (18, 21, 40, 80)
DATASETS = 400
RESAMPLES = 2000


@dataclass(frozen=True, kw_only=True)
class Calibration:
    """What the interval did over many datasets of one size.

    Attributes:
        units: How many units each dataset scores.
        datasets: How many datasets were drawn for each rate.
        resamples: Resamples per interval.
        coverage: Share of datasets with a true reduction whose interval covered it.
        false_positive: Share of datasets with no true difference whose interval excluded zero.
        one_sided_false_positive: Share of those whose whole interval lay above zero — the error
            the endpoint's confirmation rule is exposed to.
    """

    units: int
    datasets: int
    resamples: int
    coverage: float
    false_positive: float
    one_sided_false_positive: float

    COLUMNS = (
        "units",
        "datasets",
        "resamples",
        "coverage",
        "false_positive",
        "one_sided_false_positive",
    )

    def record(self) -> dict[str, str]:
        return {
            "units": str(self.units),
            "datasets": str(self.datasets),
            "resamples": str(self.resamples),
            "coverage": repr(self.coverage),
            "false_positive": repr(self.false_positive),
            "one_sided_false_positive": repr(self.one_sided_false_positive),
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            units=int(record["units"]),
            datasets=int(record["datasets"]),
            resamples=int(record["resamples"]),
            coverage=float(record["coverage"]),
            false_positive=float(record["false_positive"]),
            one_sided_false_positive=float(record["one_sided_false_positive"]),
        )


def calibrate(units: int, *, datasets: int = DATASETS, resamples: int = RESAMPLES) -> Calibration:
    """The rates over ``datasets`` synthetic datasets of ``units`` units each."""
    bootstrap = PairedUnitBootstrap(resamples=resamples, seed=1)
    shifted = KnownAnswer(control_rmse=CONTROL_RMSE, candidate_rmse=CANDIDATE_RMSE, units=units)
    null = KnownAnswer(control_rmse=CONTROL_RMSE, candidate_rmse=CONTROL_RMSE, units=units)
    covered = 0
    for seed in range(datasets):
        interval = bootstrap.compare(shifted.paired(seed=seed)).interval
        covered += interval.low <= shifted.true_reduction <= interval.high
    excluded, above = 0, 0
    for seed in range(datasets):
        interval = bootstrap.compare(null.paired(seed=seed)).interval
        excluded += interval.excludes_zero
        above += interval.above_zero
    return Calibration(
        units=units,
        datasets=datasets,
        resamples=resamples,
        coverage=covered / datasets,
        false_positive=excluded / datasets,
        one_sided_false_positive=above / datasets,
    )


def write(rows: Sequence[Calibration], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CALIBRATION).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=Calibration.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.record() for row in rows)
    return directory / CALIBRATION


def read(directory: Path) -> tuple[Calibration, ...]:
    """The calibration stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CALIBRATION).is_file():
        raise SystemExit(f"{directory} holds no {CALIBRATION}; nothing to render")
    with (directory / CALIBRATION).open(newline="", encoding="utf-8") as handle:
        return tuple(Calibration.parse(row) for row in csv.DictReader(handle))


def render(rows: Sequence[Calibration]) -> str:
    """The note's table: one row per unit count, rates as percentages."""
    return "\n\n".join(
        [
            dated_heading(),
            f"Known answer: control {CONTROL_RMSE:g}, candidate {CANDIDATE_RMSE:g} (a true "
            f"reduction of {CONTROL_RMSE - CANDIDATE_RMSE:g}) or equal; a 95 % percentile "
            f"interval over {rows[0].resamples:,} resamples; {rows[0].datasets} datasets per rate.",
            table(
                (
                    "Units",
                    "Coverage of the true reduction",
                    "Zero excluded (two-sided)",
                    "Whole interval above zero",
                ),
                (
                    (
                        str(row.units),
                        f"{row.coverage:.1%}",
                        f"{row.false_positive:.1%}",
                        f"{row.one_sided_false_positive:.1%}",
                    )
                    for row in rows
                ),
            ),
        ]
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="directory the CSV is written to")
    parser.add_argument(
        "--report-only", type=Path, help="render the table from a directory written earlier"
    )
    parser.add_argument("--datasets", type=int, default=DATASETS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(read(arguments.report_only)))
        return
    if arguments.out is None:
        raise SystemExit("either --out DIR or --report-only DIR is required")
    rows = [
        calibrate(units, datasets=arguments.datasets, resamples=arguments.resamples)
        for units in UNIT_COUNTS
    ]
    write(rows, arguments.out)
    print(render(rows))


if __name__ == "__main__":
    main()
