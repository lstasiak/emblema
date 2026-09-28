"""How well the registered interval over stays keeps its word on a binary task.

A comparison on a binary task is a paired bootstrap of the difference in the area under the ROC
curve, over units drawn in two strata, the repeats of a side pooled by the mean of their areas.
This measures that interval on the project's known-answer generator at the size of the
intensive-care task's validation side: how often a 95 % interval covers a true gain in area,
and how often it excludes, or lies wholly above, a true difference of zero. Two regimes are kept
apart: repeats that differ only unit by unit, which is what resampling units can see, and
repeats whose whole area moves from seed to seed, which it cannot. The numbers are written to
CSV first and rendered from that file, so the table can be reshaped without drawing again.

    uv run scripts/ranking_calibration_report.py --out DIR [--workers N]
    uv run scripts/ranking_calibration_report.py --report-only DIR
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Self

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.synthetic.known_ranking import KnownRanking
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from scripts.reporting import dated_heading, table

CALIBRATION = "ranking_calibration.csv"
# The validation side of the intensive-care task: set B's stays with a window, and its deaths.
UNITS = 3994
POSITIVES = 568
LEVELS = (0.70, 0.85)
GAIN = 0.02
# (repeats, spread of a repeat's area): one repeat as the interval's own case, then the five
# seeds a cell of the grid runs under, with and without a seed that moves the whole area.
REGIMES = ((1, 0.0), (5, 0.0), (5, 0.01), (5, 0.03))
DATASETS = 400
RESAMPLES = 1000


@dataclass(frozen=True, kw_only=True)
class Setting:
    """One known answer the interval is read on.

    Attributes:
        level: The control's area.
        repeats: How many repeats each side pools.
        spread: The standard deviation of one repeat's area around its side's.
    """

    level: float
    repeats: int
    spread: float

    def known(self, gain: float, *, units: int, positives: int) -> KnownRanking:
        return KnownRanking(
            control_auroc=self.level,
            candidate_auroc=self.level + gain,
            units=units,
            positives=positives,
            repeat_spread=self.spread,
        )


@dataclass(frozen=True, kw_only=True)
class Tally:
    """What one batch of datasets did: counts, and the seconds its comparisons took."""

    covered: int
    excluded: int
    above: int
    seconds: float

    def __add__(self, other: Self) -> Self:
        return type(self)(
            covered=self.covered + other.covered,
            excluded=self.excluded + other.excluded,
            above=self.above + other.above,
            seconds=self.seconds + other.seconds,
        )


@dataclass(frozen=True, kw_only=True)
class RankingCalibration:
    """What the interval did over many datasets of one setting.

    Attributes:
        level: The control's area.
        gain: The true gain in area the coverage is read on.
        repeats: How many repeats each side pools.
        spread: The standard deviation of one repeat's area around its side's.
        units: How many stays each dataset scores.
        positives: How many of them hold the positive outcome.
        datasets: How many datasets were drawn for each rate.
        resamples: Resamples per interval.
        coverage: Share of datasets with the true gain whose interval covered it.
        false_positive: Share of datasets with no true difference whose interval excluded zero.
        one_sided_false_positive: Share of those whose whole interval lay above zero — the
            error a confirmation is exposed to.
        seconds: Mean wall time of one comparison at these resamples, on one process.
    """

    level: float
    gain: float
    repeats: int
    spread: float
    units: int
    positives: int
    datasets: int
    resamples: int
    coverage: float
    false_positive: float
    one_sided_false_positive: float
    seconds: float

    COLUMNS = (
        "level",
        "gain",
        "repeats",
        "spread",
        "units",
        "positives",
        "datasets",
        "resamples",
        "coverage",
        "false_positive",
        "one_sided_false_positive",
        "seconds",
    )

    def record(self) -> dict[str, str]:
        return {column: repr(getattr(self, column)) for column in self.COLUMNS}

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            level=float(record["level"]),
            gain=float(record["gain"]),
            repeats=int(record["repeats"]),
            spread=float(record["spread"]),
            units=int(record["units"]),
            positives=int(record["positives"]),
            datasets=int(record["datasets"]),
            resamples=int(record["resamples"]),
            coverage=float(record["coverage"]),
            false_positive=float(record["false_positive"]),
            one_sided_false_positive=float(record["one_sided_false_positive"]),
            seconds=float(record["seconds"]),
        )


@dataclass(frozen=True, kw_only=True)
class Batch:
    """A share of one setting's datasets, which one process draws and reads.

    Attributes:
        known: What the datasets are drawn from.
        repeats: How many repeats each side pools.
        seeds: The seeds of this batch's datasets.
        resamples: Resamples per interval.
    """

    known: KnownRanking
    repeats: int
    seeds: range
    resamples: int

    def tally(self) -> Tally:
        """How the interval did on this batch's datasets."""
        bootstrap = PairedUnitBootstrap(resamples=self.resamples, seed=1)
        covered = excluded = above = 0
        seconds = 0.0
        for seed in self.seeds:
            paired = self.known.repeats(self.repeats, seed=seed)
            started = time.perf_counter()
            interval = bootstrap.compare(paired).interval
            seconds += time.perf_counter() - started
            covered += interval.low <= self.known.true_reduction <= interval.high
            excluded += interval.excludes_zero
            above += interval.above_zero
        return Tally(covered=covered, excluded=excluded, above=above, seconds=seconds)


def calibrate(
    settings: Sequence[Setting],
    *,
    units: int = UNITS,
    positives: int = POSITIVES,
    gain: float = GAIN,
    datasets: int = DATASETS,
    resamples: int = RESAMPLES,
    workers: int = 1,
) -> tuple[RankingCalibration, ...]:
    """The rates of every setting, its datasets split into batches over ``workers`` processes."""
    keys: list[tuple[int, bool]] = []
    batches: list[Batch] = []
    for index, setting in enumerate(settings):
        for shifted, gain_drawn in ((True, gain), (False, 0.0)):
            known = setting.known(gain_drawn, units=units, positives=positives)
            for start in range(workers):
                keys.append((index, shifted))
                batches.append(
                    Batch(
                        known=known,
                        repeats=setting.repeats,
                        seeds=range(start, datasets, workers),
                        resamples=resamples,
                    )
                )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tallies = list(pool.map(Batch.tally, batches))
    else:
        tallies = [batch.tally() for batch in batches]
    totals: dict[tuple[int, bool], Tally] = {}
    for key, found in zip(keys, tallies, strict=True):
        totals[key] = totals[key] + found if key in totals else found
    return tuple(
        RankingCalibration(
            level=setting.level,
            gain=gain,
            repeats=setting.repeats,
            spread=setting.spread,
            units=units,
            positives=positives,
            datasets=datasets,
            resamples=resamples,
            coverage=totals[index, True].covered / datasets,
            false_positive=totals[index, False].excluded / datasets,
            one_sided_false_positive=totals[index, False].above / datasets,
            seconds=(totals[index, True].seconds + totals[index, False].seconds) / (2 * datasets),
        )
        for index, setting in enumerate(settings)
    )


def write(rows: Sequence[RankingCalibration], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / CALIBRATION).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RankingCalibration.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.record() for row in rows)
    return directory / CALIBRATION


def read(directory: Path) -> tuple[RankingCalibration, ...]:
    """The calibration stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / CALIBRATION).is_file():
        raise SystemExit(f"{directory} holds no {CALIBRATION}; nothing to render")
    with (directory / CALIBRATION).open(newline="", encoding="utf-8") as handle:
        return tuple(RankingCalibration.parse(row) for row in csv.DictReader(handle))


def render(rows: Sequence[RankingCalibration]) -> str:
    """The note's table: one row per setting, rates as percentages."""
    first = rows[0]
    return "\n\n".join(
        [
            dated_heading(),
            f"Known answer: {first.units:,} stays, {first.positives} positive; a true gain of "
            f"{first.gain:g} in area, or none; a 95 % percentile interval over "
            f"{first.resamples:,} resamples in two strata; {first.datasets} datasets per rate.",
            table(
                (
                    "Control's area",
                    "Repeats",
                    "Spread of a repeat's area",
                    "Coverage of the true gain",
                    "Zero excluded (two-sided)",
                    "Whole interval above zero",
                    "Seconds a comparison",
                ),
                (
                    (
                        f"{row.level:.2f}",
                        str(row.repeats),
                        f"{row.spread:g}",
                        f"{row.coverage:.1%}",
                        f"{row.false_positive:.1%}",
                        f"{row.one_sided_false_positive:.1%}",
                        f"{row.seconds:.2f}",
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
    parser.add_argument("--workers", type=int, default=1, help="processes the datasets share")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(read(arguments.report_only)))
        return
    if arguments.out is None:
        raise SystemExit("either --out DIR or --report-only DIR is required")
    if arguments.workers < 1:
        raise SystemExit(f"--workers must be at least one, got {arguments.workers}")
    settings = [
        Setting(level=level, repeats=repeats, spread=spread)
        for level in LEVELS
        for repeats, spread in REGIMES
    ]
    rows = calibrate(
        settings,
        units=UNITS,
        positives=POSITIVES,
        datasets=arguments.datasets,
        resamples=arguments.resamples,
        workers=arguments.workers,
    )
    write(rows, arguments.out)
    print(render(rows))


if __name__ == "__main__":
    main()
