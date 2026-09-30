"""How well the registered interval over stays keeps its word on a binary task.

A comparison on a binary task is a paired bootstrap of the difference in the area under the ROC
curve, over units drawn in two strata, the repeats of a side pooled by the mean of their areas.
This measures that interval on the project's known-answer generator at the size of the
intensive-care task's validation side: how often a 95 % interval covers a true gain in area,
and how often it excludes, or lies wholly above, a true difference of zero. Two regimes are kept
apart: repeats that differ only unit by unit, which is what resampling units can see, and
repeats whose whole area moves from seed to seed, which it cannot. A second suite asks whether
the interval keeps its level where the generator's defaults stop describing a model: a lower
area, two sides that share more or less of their answers, and answers that tie.

The decomposition explains the coverage under a spread of repeats rather than measuring it
again: the spread over datasets of the point difference, with and without the spread, gives the
two parts of its variance, and the coverage a normal interval of the first part would keep
against both. Set beside the measured coverage, it says whether that mechanism accounts for it.

The numbers are written to CSV first and rendered from those files, so the tables can be
reshaped without drawing again.

    uv run scripts/ranking_calibration_report.py --out DIR [--suite robustness] [--workers N]
    uv run scripts/ranking_calibration_report.py --out DIR --decompose [--workers N]
    uv run scripts/ranking_calibration_report.py --report-only DIR
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from statistics import NormalDist, stdev
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
DECOMPOSITION = "decomposition.csv"
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
# The generator's defaults: half an answer's variance within an outcome is the stay's.
SHARED = 0.5
# Seeds of the decomposition's datasets start here, clear of the calibration's.
DECOMPOSITION_SEEDS = 10_000


@dataclass(frozen=True, kw_only=True)
class Setting:
    """One known answer the interval is read on.

    Attributes:
        level: The control's area.
        repeats: How many repeats each side pools.
        spread: The standard deviation of one repeat's area around its side's.
        shared: The share of an answer's variance within an outcome that is the stay's.
        levels: How many distinct answers each side gives; zero for answers that never tie.
    """

    level: float
    repeats: int
    spread: float
    shared: float = SHARED
    levels: int = 0

    def known(self, gain: float, *, units: int, positives: int) -> KnownRanking:
        return KnownRanking(
            control_auroc=self.level,
            candidate_auroc=self.level + gain,
            units=units,
            positives=positives,
            shared=self.shared,
            repeat_spread=self.spread,
            answer_levels=self.levels,
        )


def suite(name: str) -> tuple[Setting, ...]:
    """The settings a suite reads the interval on.

    Raises:
        SystemExit: If no suite has that name.
    """
    if name == "registered":
        return tuple(
            Setting(level=level, repeats=repeats, spread=spread)
            for level in LEVELS
            for repeats, spread in REGIMES
        )
    if name == "robustness":
        # Five repeats and no spread, the case the registered suite found at its level, moved one
        # assumption at a time: a model barely above chance, two sides nearly independent or
        # nearly the same, and answers in twenty or in five distinct values.
        return (
            Setting(level=0.60, repeats=5, spread=0.0),
            Setting(level=0.80, repeats=5, spread=0.0, shared=0.2),
            Setting(level=0.80, repeats=5, spread=0.0, shared=0.9),
            Setting(level=0.80, repeats=5, spread=0.0, levels=20),
            Setting(level=0.80, repeats=5, spread=0.0, levels=5),
        )
    raise SystemExit(f"no suite named {name!r}; the suites are registered and robustness")


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
        gain: The true gain in area the coverage is read on, as the answers can reach it.
        repeats: How many repeats each side pools.
        spread: The standard deviation of one repeat's area around its side's.
        shared: The share of an answer's variance within an outcome that is the stay's.
        levels: How many distinct answers each side gives; zero for answers that never tie.
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
    shared: float
    levels: int
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
        "shared",
        "levels",
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
            # A file written before the two columns existed was drawn under the defaults.
            shared=float(record.get("shared", SHARED)),
            levels=int(record.get("levels", 0)),
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
            gain=setting.known(gain, units=units, positives=positives).true_reduction,
            repeats=setting.repeats,
            spread=setting.spread,
            shared=setting.shared,
            levels=setting.levels,
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


@dataclass(frozen=True, kw_only=True)
class Draws:
    """A share of one setting's datasets whose point difference one process computes.

    Attributes:
        known: What the datasets are drawn from.
        repeats: How many repeats each side pools.
        seeds: The seeds of this share's datasets.
    """

    known: KnownRanking
    repeats: int
    seeds: range

    def reductions(self) -> list[float]:
        return [self.known.repeats(self.repeats, seed=seed).reduction for seed in self.seeds]


@dataclass(frozen=True, kw_only=True)
class Decomposition:
    """The two parts of the point difference's variance under one spread of repeats.

    Attributes:
        level: The control's area; the candidate's is the same, so the true difference is zero.
        repeats: How many repeats each side pools.
        spread: The standard deviation of one repeat's area around its side's.
        datasets: How many datasets each spread was read on.
        over_stays: The difference's standard deviation over datasets without the spread: what
            resampling stays can see.
        over_both: Its standard deviation with the spread.
    """

    level: float
    repeats: int
    spread: float
    datasets: int
    over_stays: float
    over_both: float

    COLUMNS = ("level", "repeats", "spread", "datasets", "over_stays", "over_both")

    @property
    def over_seeds(self) -> float:
        """The part the spread adds, as a standard deviation."""
        return sqrt(max(self.over_both**2 - self.over_stays**2, 0.0))

    @property
    def expected_over_seeds(self) -> float:
        """What the spread should add: the mean of ``repeats`` independent draws, on two sides."""
        return self.spread * sqrt(2.0 / self.repeats)

    @property
    def predicted_coverage(self) -> float:
        """The coverage of a normal 95 % interval as wide as the stays alone make it."""
        standard = NormalDist()
        return 2.0 * standard.cdf(standard.inv_cdf(0.975) * self.over_stays / self.over_both) - 1.0

    def record(self) -> dict[str, str]:
        return {column: repr(getattr(self, column)) for column in self.COLUMNS}

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            level=float(record["level"]),
            repeats=int(record["repeats"]),
            spread=float(record["spread"]),
            datasets=int(record["datasets"]),
            over_stays=float(record["over_stays"]),
            over_both=float(record["over_both"]),
        )


def decompose(
    settings: Sequence[Setting],
    *,
    units: int = UNITS,
    positives: int = POSITIVES,
    datasets: int = DATASETS,
    workers: int = 1,
) -> tuple[Decomposition, ...]:
    """Each spread's settings split into the variance over stays and the variance it adds.

    Every setting with a spread is read against the same setting without it, on datasets with no
    true difference, seeded apart from the calibration's so that neither borrows the other's.
    """
    spread = [setting for setting in settings if setting.spread]
    steady = {setting: replace(setting, spread=0.0) for setting in spread}
    readings = list(dict.fromkeys([*steady.values(), *spread]))
    keys: list[int] = []
    shares: list[Draws] = []
    for index, setting in enumerate(readings):
        known = setting.known(0.0, units=units, positives=positives)
        for start in range(workers):
            keys.append(index)
            shares.append(
                Draws(
                    known=known,
                    repeats=setting.repeats,
                    seeds=range(
                        DECOMPOSITION_SEEDS + start, DECOMPOSITION_SEEDS + datasets, workers
                    ),
                )
            )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            drawn = list(pool.map(Draws.reductions, shares))
    else:
        drawn = [share.reductions() for share in shares]
    pooled: dict[int, list[float]] = {}
    for key, reductions in zip(keys, drawn, strict=True):
        pooled.setdefault(key, []).extend(reductions)
    deviation = {setting: stdev(pooled[index]) for index, setting in enumerate(readings)}
    return tuple(
        Decomposition(
            level=setting.level,
            repeats=setting.repeats,
            spread=setting.spread,
            datasets=datasets,
            over_stays=deviation[steady[setting]],
            over_both=deviation[setting],
        )
        for setting in spread
    )


def write(rows: Sequence[RankingCalibration], directory: Path) -> Path:
    return _write(
        CALIBRATION, RankingCalibration.COLUMNS, [row.record() for row in rows], directory
    )


def write_decomposition(rows: Sequence[Decomposition], directory: Path) -> Path:
    return _write(DECOMPOSITION, Decomposition.COLUMNS, [row.record() for row in rows], directory)


def _write(
    name: str, columns: Sequence[str], records: Sequence[dict[str, str]], directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return directory / name


def read(directory: Path) -> tuple[RankingCalibration, ...]:
    """The calibration stored under ``directory``, or none."""
    return tuple(RankingCalibration.parse(row) for row in _read(CALIBRATION, directory))


def read_decomposition(directory: Path) -> tuple[Decomposition, ...]:
    """The decomposition stored under ``directory``, or none."""
    return tuple(Decomposition.parse(row) for row in _read(DECOMPOSITION, directory))


def _read(name: str, directory: Path) -> list[dict[str, str]]:
    if not (directory / name).is_file():
        return []
    with (directory / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(rows: Sequence[RankingCalibration]) -> str:
    """The note's table: one row per setting, rates as percentages."""
    first = rows[0]
    return "\n\n".join(
        [
            f"Known answer: {first.units:,} stays, {first.positives} positive; a true gain in "
            f"area, or none; a 95 % percentile interval over {first.resamples:,} resamples in "
            f"two strata; {first.datasets} datasets per rate.",
            table(
                (
                    "Control's area",
                    "Repeats",
                    "Spread of a repeat's area",
                    "Shared",
                    "Levels",
                    "True gain",
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
                        f"{row.shared:g}",
                        str(row.levels) if row.levels else "—",
                        f"{row.gain:.4f}",
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


def render_decomposition(
    rows: Sequence[Decomposition], measured: Sequence[RankingCalibration]
) -> str:
    """The decomposition's table, with the coverage the calibration measured where it has one."""
    coverage = {
        (row.level, row.repeats, row.spread): row.coverage
        for row in measured
        if row.shared == SHARED and not row.levels
    }
    return "\n\n".join(
        [
            f"No true difference; {rows[0].datasets} datasets per spread; the part over seeds "
            "expected from the spread is the spread times the square root of two over the repeats.",
            table(
                (
                    "Control's area",
                    "Repeats",
                    "Spread of a repeat's area",
                    "SD over stays",
                    "SD over both",
                    "SD over seeds (expected)",
                    "Coverage predicted",
                    "Coverage measured",
                ),
                (
                    (
                        f"{row.level:.2f}",
                        str(row.repeats),
                        f"{row.spread:g}",
                        f"{row.over_stays:.4f}",
                        f"{row.over_both:.4f}",
                        f"{row.over_seeds:.4f} ({row.expected_over_seeds:.4f})",
                        f"{row.predicted_coverage:.1%}",
                        (
                            f"{coverage[key]:.1%}"
                            if (key := (row.level, row.repeats, row.spread)) in coverage
                            else "—"
                        ),
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
        "--report-only", type=Path, help="render the tables from a directory written earlier"
    )
    parser.add_argument("--suite", default="registered", help="registered or robustness")
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="split the registered suite's spreads into their parts instead of calibrating",
    )
    parser.add_argument("--datasets", type=int, default=DATASETS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--workers", type=int, default=1, help="processes the datasets share")
    return parser.parse_args(argv)


def report(directory: Path) -> str:
    """Every table stored under ``directory``, under one dated heading.

    Raises:
        SystemExit: If nothing is stored there.
    """
    calibration = read(directory)
    decomposition = read_decomposition(directory)
    if not calibration and not decomposition:
        raise SystemExit(
            f"{directory} holds no {CALIBRATION} or {DECOMPOSITION}; nothing to render"
        )
    parts = [dated_heading()]
    if calibration:
        parts.append(render(calibration))
    if decomposition:
        parts.append(render_decomposition(decomposition, calibration))
    return "\n\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(report(arguments.report_only))
        return
    if arguments.out is None:
        raise SystemExit("either --out DIR or --report-only DIR is required")
    if arguments.workers < 1:
        raise SystemExit(f"--workers must be at least one, got {arguments.workers}")
    settings = suite(arguments.suite)
    if arguments.decompose:
        write_decomposition(
            decompose(
                settings,
                units=UNITS,
                positives=POSITIVES,
                datasets=arguments.datasets,
                workers=arguments.workers,
            ),
            arguments.out,
        )
    else:
        rows = calibrate(
            settings,
            units=UNITS,
            positives=POSITIVES,
            datasets=arguments.datasets,
            resamples=arguments.resamples,
            workers=arguments.workers,
        )
        write(rows, arguments.out)
    print(report(arguments.out))


if __name__ == "__main__":
    main()
