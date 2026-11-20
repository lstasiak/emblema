r"""Local statistical forecasters' answer for each cell's hidden hour, fitted on its own hours.

Each scored cell's 47 hourly means, as the answers report wrote them for the published
forecaster, are handed to four forecasters fitted on those hours alone — AutoETS, AutoARIMA,
AutoTheta and the seasonal naive answer of ``statsforecast`` — so each adapts to its own window,
where the ridge per channel was fitted once on the training months. A cell's season is the
period the same-phase answer chose from the same hours (the ``lag`` column of
``answers-<label>.csv``), and one where that period is shorter than two hours or longer than
half the context. The forecasters take no missing hour, so an hour the channel did not read
carries the hour before it, and the hours before its first reading take that reading.

The forecasters are not dependencies of this package, so this runs in its own environment:

    uv run --isolated --no-project --python 3.12 --with statsforecast==2.1.1 \\
        scripts/local_forecaster_answers.py --context DIR/context-<label>.csv \\
        --answers DIR/answers-<label>.csv --out DIR/local-<label>.csv

Writes one row per cell: window, axis, channel and each forecaster's answer for the hour's mean.
"""

import argparse
import csv
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_answers import ContextRow, read_context  # noqa: E402

Key = tuple[int, str, int]
# The forecasters by the column each writes, in the order the file holds them.
FORECASTERS = ("auto_ets", "auto_arima", "auto_theta", "seasonal_naive")
COLUMNS = ("window", "axis", "channel", *(f"mean_{name}" for name in FORECASTERS))
# Fits a batch of series sharing a season: ``[series, hours]`` in, one answer per forecaster out.
Fit = Callable[[NDArray[np.float64], int], Mapping[str, NDArray[np.float64]]]


def read_lags(path: Path) -> dict[Key, int]:
    """Each scored cell's same-phase period from an answers file; a blank lag reads as one."""
    with path.open(newline="") as handle:
        return {
            (int(row["window"]), row["axis"], int(row["channel"])): int(row["lag"] or 1)
            for row in csv.DictReader(handle)
        }


def filled(means: NDArray[np.float32] | NDArray[np.float64]) -> NDArray[np.float64]:
    """The hours with each missing one carrying the hour before it, the first reading backwards.

    Raises:
        ValueError: If no hour holds a reading.
    """
    values = np.asarray(means, dtype=np.float64)
    read = np.isfinite(values)
    if not read.any():
        raise ValueError("a context with no hour read cannot be filled")
    last = np.where(read, np.arange(len(values)), 0)
    np.maximum.accumulate(last, out=last)
    out = values[last]
    first = int(np.argmax(read))
    out[:first] = values[first]
    return out


def season_of(lag: int, hours: int) -> int:
    """The season a cell's forecasters use: its period when it fits twice in the context."""
    return lag if 2 <= lag <= hours // 2 else 1


def by_season(rows: Sequence[ContextRow], lags: Mapping[Key, int]) -> dict[int, list[ContextRow]]:
    """The cells grouped by the season their forecasters use.

    Raises:
        ValueError: If a cell has no lag in the answers, so the two files name other cells.
    """
    groups: dict[int, list[ContextRow]] = {}
    for row in rows:
        key = (row.window, row.axis, row.channel)
        if key not in lags:
            raise ValueError(f"window {row.window} channel {row.channel} has no answer row")
        groups.setdefault(season_of(lags[key], len(row.means)), []).append(row)
    return groups


def answer(
    rows: Sequence[ContextRow],
    lags: Mapping[Key, int],
    fit: Fit,
    *,
    say: Callable[[str], None] = lambda _: None,
) -> dict[Key, dict[str, float]]:
    """Every cell's answer from each forecaster, the cells of one season fitted together.

    Raises:
        ValueError: If ``fit`` answers another number of cells or forecasters than asked, or an
            answer that is not finite.
    """
    answered: dict[Key, dict[str, float]] = {}
    started = time.monotonic()
    for season, group in sorted(by_season(rows, lags).items()):
        series = np.stack([filled(row.means) for row in group])
        outputs = fit(series, season)
        if set(outputs) != set(FORECASTERS):
            raise ValueError(f"the fit answered {sorted(outputs)}, not {list(FORECASTERS)}")
        for name, values in outputs.items():
            if len(values) != len(group):
                raise ValueError(f"{name} answered {len(values)} cells for {len(group)} handed")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} answered a value that is not finite at season {season}")
        for k, row in enumerate(group):
            answered[(row.window, row.axis, row.channel)] = {
                name: float(outputs[name][k]) for name in FORECASTERS
            }
        say(f"season {season}: {len(group)} cells, {time.monotonic() - started:.0f} s")
    return answered


def write_answers(
    path: Path, rows: Sequence[ContextRow], answered: Mapping[Key, Mapping[str, float]]
) -> None:
    """One row per cell, in the context file's order."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            values = answered[(row.window, row.axis, row.channel)]
            writer.writerow(
                (row.window, row.axis, row.channel, *(f"{values[n]:.6g}" for n in FORECASTERS))
            )


def statsforecast_fit(jobs: int) -> Fit:  # pragma: no cover - needs the forecasters
    """The four forecasters of ``statsforecast``, every series fitted on its own."""
    import pandas as pd  # type: ignore[import-untyped,unused-ignore]
    import statsforecast  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import statsforecast.models as sf  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]

    names = {
        "AutoETS": "auto_ets",
        "AutoARIMA": "auto_arima",
        "AutoTheta": "auto_theta",
        "SeasonalNaive": "seasonal_naive",
    }

    def fit(series: NDArray[np.float64], season: int) -> dict[str, NDArray[np.float64]]:
        count, hours = series.shape
        frame = pd.DataFrame(
            {
                "unique_id": np.repeat(np.arange(count), hours),
                "ds": np.tile(np.arange(1, hours + 1), count),
                "y": series.ravel(),
            }
        )
        models = [
            sf.AutoETS(season_length=season),
            sf.AutoARIMA(season_length=season),
            sf.AutoTheta(season_length=season),
            sf.SeasonalNaive(season_length=season),
        ]
        engine = statsforecast.StatsForecast(models=models, freq=1, n_jobs=jobs)
        forecasts = engine.forecast(df=frame, h=1)
        if "unique_id" not in forecasts.columns:
            forecasts = forecasts.reset_index()
        forecasts = forecasts.sort_values("unique_id")
        if not np.array_equal(forecasts["unique_id"].to_numpy(), np.arange(count)):
            raise ValueError("the forecasts name other series than the ones handed")
        return {names[name]: forecasts[name].to_numpy(dtype=np.float64) for name in names}

    return fit


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the forecasters
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True, help="the cells' same-phase lags")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=-1, help="processes; all cores by default")
    arguments = parser.parse_args(argv)

    rows = read_context(arguments.context)
    lags = read_lags(arguments.answers)
    started = time.monotonic()
    answered = answer(
        rows,
        lags,
        statsforecast_fit(arguments.jobs),
        say=lambda text: print(text, file=sys.stderr),
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write_answers(arguments.out, rows, answered)
    print(f"{len(answered)} cells answered in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":  # pragma: no cover
    main()
