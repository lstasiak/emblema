r"""The fine-tuned forecaster's pilot read: its answers against the trees and the zero-shot model.

The pilot reads the cells of turbines read as new on their first month (days 2, 7 and 14
pooled), as ``fleet_cold_start_reading.py`` reads them, by the same statistic: per window, one
turbine's hour, the mean over its temperatures of the squared error; per comparison, the
geometric mean over windows of the answer's window loss over the bar's, the bar at its best
member, with an interval from whole starts drawn again. Each comparison reads the cells its
first answer gives. Nothing here is a verdict.

Each ``--compare OF AGAINST`` names an answer, and an answer or one of the bars:

- ``normal behaviour without the farm``: the fleet reading's bar of answers from the new
  turbine alone;
- ``normal behaviour without the farm, other farm's trees``: the same with the trees fitted on
  the other farm before 2020, the best a farm read as new has without any history of its own.

A reading names further bars and answers of its own:

- ``--bar NAME MEMBER...``: a bar of the answers named, read at its best member, as the
  tuned trees and the trees corrected by the turbine's own hours are; a member that names a
  bar above stands for that bar's members;
- ``--mean NAME ANSWER...``: an answer that is the mean of several, as of two seeds' fits;
- ``--by-day``: each comparison on each verdict day too, beside the month pooled.

    uv run scripts/fleet_adaptation_reading.py --cells DIR --farm kelmarsh \\
        --kinds answers local chronos2 trees-pilot adapted_fleet \\
        --compare adapted_fleet_nbm fleet_nbm_trees_folds_until2020 ... --out DIR

Reads ``<kind>-<farm>-N<days>.csv`` for each kind and verdict day from ``--cells``; writes
``comparisons.csv`` and ``losses.csv``.
"""

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cold_start_reading import Day, joined, pooled  # noqa: E402
from scripts.fleet_cold_start_cells import VERDICT_DAYS  # noqa: E402
from scripts.fleet_cold_start_reading import (  # noqa: E402
    COMPARISON_COLUMNS,
    LOSS_COLUMNS,
    NBM_ALONE,
    Comparison,
    comparison_rows,
    loss_rows,
)
from scripts.history_forecast_reading import markdown, write  # noqa: E402

WITHOUT_THE_FARM = "normal behaviour without the farm"
BARS = {
    WITHOUT_THE_FARM: NBM_ALONE,
    f"{WITHOUT_THE_FARM}, other farm's trees": (*NBM_ALONE, "other_farm_nbm_trees_until2020"),
}


def verdict_days(cells: Path, farm: str, kinds: Sequence[str]) -> list[Day]:
    """Each verdict day's cells of the farm, every kind's answers joined.

    Raises:
        ValueError: If a kind's file of a verdict day is missing.
    """
    days = []
    for number in VERDICT_DAYS:
        files = [cells / f"{kind}-{farm}-N{number}.csv" for kind in kinds]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise ValueError(f"{farm}-N{number}: {missing} not written")
        days.append(joined(farm, str(number), files))
    return days


def with_means(day: Day, means: Mapping[str, Sequence[str]]) -> Day:
    """The day with each named mean of several answers as an answer, NaN where one gave none.

    Raises:
        ValueError: If a mean takes an answer no file gives, or the name of one a file gives.
    """
    answers = dict(day.answers)
    for name, parts in means.items():
        absent = sorted(set(parts) - set(day.answers))
        if absent or name in day.answers:
            raise ValueError(f"the mean {name} cannot be taken: {absent or 'a file answers it'}")
        answers[name] = np.mean([day.answers[part] for part in parts], axis=0)
    return replace(day, answers=answers)


def widened(members: Sequence[str]) -> tuple[str, ...]:
    """The members, one of ``BARS`` standing for its own.

    Raises:
        ValueError: If there is none: a bar of nothing would leave its comparisons unread.
    """
    if not members:
        raise ValueError("a bar needs a member")
    return tuple(name for member in members for name in BARS.get(member, (member,)))


def compared(
    day: Day,
    pairs: Sequence[tuple[str, str]],
    *,
    resamples: int,
    seed: int,
    bars: Mapping[str, tuple[str, ...]] | None = None,
    reading: str = "first month",
) -> list[dict[str, object]]:
    """Each pair's row on the cells its first answer gives; ``bars`` are read beside ``BARS``.

    Raises:
        ValueError: If a pair names an answer no file gives, a bar lacks a member it adds, or
            a bar takes the name of another.
    """
    if bars and set(bars) & set(BARS):
        raise ValueError(f"{sorted(set(bars) & set(BARS))} name bars already")
    every = {**BARS, **(bars or {})}
    named = [name for pair in pairs for name in pair if name not in every]
    # A member a bar adds to the fleet reading's must be there, or the bar would quietly shrink.
    added = [m for pair in pairs if pair[1] in every for m in every[pair[1]] if m not in NBM_ALONE]
    absent = sorted({name for name in (*named, *added) if name not in day.answers})
    if absent:
        raise ValueError(f"no file answers {absent}")
    comparisons = [Comparison(of, against, scope=of) for of, against in pairs]
    return comparison_rows(
        day,
        comparisons,
        reading=reading,
        levels=False,
        least_gain=1.0,
        resamples=resamples,
        seed=seed,
        bars=every,
    )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--farm", required=True)
    parser.add_argument("--kinds", nargs="+", required=True)
    parser.add_argument(
        "--compare", nargs=2, action="append", required=True, metavar=("OF", "AGAINST")
    )
    parser.add_argument("--bar", nargs="+", action="append", default=[], metavar="NAME MEMBER")
    parser.add_argument("--mean", nargs="+", action="append", default=[], metavar="NAME ANSWER")
    parser.add_argument("--by-day", action="store_true")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    bars = {name: widened(members) for name, *members in arguments.bar}
    means = {name: tuple(parts) for name, *parts in arguments.mean}
    days = [
        with_means(d, means) for d in verdict_days(arguments.cells, arguments.farm, arguments.kinds)
    ]
    day = pooled(days)
    pairs = [tuple(pair) for pair in arguments.compare]
    rows: list[dict[str, object]] = []
    for one in (day, *(days if arguments.by_day else ())):
        rows += compared(
            one,
            pairs,
            resamples=arguments.resamples,
            seed=arguments.seed,
            bars=bars,
            reading="first month" if one is day else f"day {one.days}",
        )
    answers = sorted({name for pair in pairs for name in pair if name not in {**BARS, **bars}})
    lost = loss_rows(day, arguments.resamples, arguments.seed, answers)
    arguments.out.mkdir(parents=True, exist_ok=True)
    write(arguments.out / "comparisons.csv", COMPARISON_COLUMNS, rows)
    write(arguments.out / "losses.csv", LOSS_COLUMNS, lost)
    print(markdown(COMPARISON_COLUMNS, rows))
    print()
    print(markdown(LOSS_COLUMNS, lost))


if __name__ == "__main__":  # pragma: no cover
    main()
