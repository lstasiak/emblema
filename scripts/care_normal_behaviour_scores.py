r"""A model of normal behaviour over the fault benchmark's excerpts, hour by hour.

The comparator the wind industry would reach for: each component temperature of an hour answered
from the hour's operating conditions (wind speed, active power, rotor speed, pitch and the
temperatures outside) and the mean power of the 3, 6 and 24 hours before, which carries what the
components have stored; never from the temperature's own past, which follows a fault as it
grows. An hour's score is how far the temperatures lie from their answers, each in the robust
deviations of its own residuals, the largest or the mean over the temperatures.

The model is fitted on the excerpt's fitted year. Its residuals over that year are taken out of
fold, the year cut into weeks dealt to six folds in turn so every fold holds every season; a
model fitted on a week would put its residuals there far below those of the scored window, and
the year's blocks, which the scored event is read beside, would look too quiet. The scored
window is answered by the model of the whole year.

Beside, descriptively, the turbine with no history: the same model fitted on the fitted years of
the farm's other turbines that are read and none of the excerpt's own rows; its residuals are
still centred and scaled over the excerpt's own year, as the event is still read beside it.

    uv run scripts/care_normal_behaviour_scores.py --root data/raw/care/CARE_To_Compare \\
        --out DIR --form trees [--farm A B C] [--event FARM NUMBER]... [--seed 1] [--jobs 8]

Writes ``hourly-<farm>-<event>.npz``, each excerpt's hours as the model reads them, anew for
every excerpt of the farms named, since a farm's model reads them all; then
``behaviour-<farm>-<event>.npz`` per excerpt, as ``care_departure_scores.py`` writes its files,
and prints per excerpt the share of the temperatures' variance the out-of-fold answers explain.
The excerpts run side by side, a fit on one thread. Refuses to run where torch is loaded, whose
OpenMP the trees' must not meet.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.domain.channels.channel_statistics import MAD_TO_DEVIATION  # noqa: E402
from scripts.care_excerpts import (  # noqa: E402
    FARMS,
    POWER,
    Event,
    Excerpt,
    Hours,
    Sensors,
    chosen,
    read_events,
    read_excerpt,
    sensors_of,
)
from scripts.cold_start_cells import Ridge  # noqa: E402
from scripts.fleet_tree_answers import MOST_ROWS, Fit, xgboost_fit  # noqa: E402

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

FAMILY = "normal_behaviour"
FLEET = "normal_behaviour_fleet"
# The hours before an hour over which the power's mean is known to the model.
LOADS = (3, 6, 24)
FOLDS = 6
HOURS_PER_WEEK = 168


@dataclass(frozen=True)
class Hourly:
    """An excerpt as a normal-behaviour model reads it.

    Attributes:
        features: ``[hours, conditions + loads]``, missing where an hour is not whole or the
            hours before it hold too little power.
        targets: ``[hours, temperatures]``, the whole hours' means.
        fitted: How many hours, from the first, the model may be fitted on.
    """

    features: Floats
    targets: Floats
    fitted: int

    @classmethod
    def of(cls, excerpt: Excerpt, hours: Hours, sensors: Sensors) -> Hourly:
        conditions = hours.readings(excerpt.values[:, excerpt.columns_at(sensors.conditions)])
        targets = hours.readings(excerpt.values[:, excerpt.columns_at(sensors.temperatures)])
        power = excerpt.values[:, excerpt.columns_at(sensors.conditions[POWER : POWER + 1])[0]]
        return cls(
            features=np.column_stack(
                [conditions.mean(axis=2), loads_before(hourly_mean(power, hours))]
            ),
            targets=targets.mean(axis=2),
            fitted=hours.fitted,
        )


def hourly_mean(values: Floats, hours: Hours) -> Floats:
    """Per hour, the mean of the rows that hold a reading, whatever their status."""
    held = ~np.isnan(values)
    count = np.bincount(hours.of_row[held], minlength=hours.count)
    total = np.bincount(hours.of_row[held], weights=values[held], minlength=hours.count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: Floats = np.where(count > 0, total / count, np.nan)
    return mean


def loads_before(power: Floats) -> Floats:
    """``[hours, loads]``: the mean power of the hours before each hour, the hour itself left out.

    Missing where fewer than half of them hold a reading. A stopped turbine's hours count, since
    the components cool through them. Each mean is summed over its own hours, so the same hours
    give the same load to the last bit wherever an excerpt starts.
    """
    loads = np.full((len(power), len(LOADS)), np.nan)
    for column, reach in enumerate(LOADS):
        before = sliding_window_view(np.concatenate([np.full(reach, np.nan), power]), reach)[:-1]
        known = (~np.isnan(before)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(np.isnan(before), 0.0, before).sum(axis=1) / known
        loads[:, column] = np.where(known * 2 >= reach, mean, np.nan)
    return loads


def folds_of(hours: int) -> Ints:
    """Each hour's fold: the weeks dealt to the folds in turn."""
    dealt: Ints = (np.arange(hours) // HOURS_PER_WEEK) % FOLDS
    return dealt


def ridge_fit(rows: Floats, targets: Floats, seed: int) -> Callable[[Floats], Floats]:
    """A ridge on the standardised rows, its penalty the one each half answers the other best by."""
    centre, spread = rows.mean(axis=0), rows.std(axis=0)
    spread = np.where(spread > 0.0, spread, 1.0)
    ridge = Ridge.fitted((rows - centre) / spread, targets)

    def answer(new: Floats) -> Floats:
        if ridge is None:
            return np.full(len(new), np.nan)
        clipped = np.clip((new - centre) / spread, ridge.low, ridge.high)
        answers: Floats = clipped @ ridge.weights + ridge.intercept
        return answers

    return answer


def answers_of(hourly: Hourly, target: int, fit: Fit, seed: int) -> Floats:
    """One temperature's answer in every hour whose features are whole.

    Over the fitted hours out of fold; after them by the model of all the fitted hours.
    """
    rows, values = hourly.features, hourly.targets[:, target]
    whole = np.isfinite(rows).all(axis=1)
    known = whole & np.isfinite(values)
    known[hourly.fitted :] = False
    answers = np.full(len(values), np.nan)
    if known.sum() < 2 * FOLDS:
        return answers
    folds = folds_of(len(values))
    for fold in range(FOLDS):
        held = whole & (folds == fold)
        held[hourly.fitted :] = False
        fitted_on = known & (folds != fold)
        if held.any() and fitted_on.any():
            answers[held] = fit(rows[fitted_on], values[fitted_on], seed)(rows[held])
    after = whole.copy()
    after[: hourly.fitted] = False
    if after.any():
        answers[after] = fit(rows[known], values[known], seed)(rows[after])
    return answers


def fleet_answers(
    hourly: Hourly, others: Sequence[Hourly], target: int, fit: Fit, seed: int
) -> Floats:
    """One temperature's answer in every whole hour by a model of the other turbines' years."""
    rows = np.concatenate([other.features[: other.fitted] for other in others])
    values = np.concatenate([other.targets[: other.fitted, target] for other in others])
    known = np.isfinite(rows).all(axis=1) & np.isfinite(values)
    table = np.column_stack([rows[known], values[known]])
    # Two excerpts of a turbine can hold the same year; each of its hours is fitted on once,
    # in the order the hours were read.
    table = table[np.sort(np.unique(table, axis=0, return_index=True)[1])]
    if len(table) > MOST_ROWS:
        kept = np.random.default_rng(seed).choice(len(table), MOST_ROWS, replace=False)
        table = table[np.sort(kept)]
    answers = np.full(len(hourly.targets), np.nan)
    whole = np.isfinite(hourly.features).all(axis=1)
    if len(table) >= 2 * FOLDS and whole.any():
        answers[whole] = fit(table[:, :-1], table[:, -1], seed)(hourly.features[whole])
    return answers


def deviations(answers: Floats, values: Floats, fitted: int) -> Floats:
    """How far a temperature lies from its answer.

    In the robust deviations of the fitted hours' residuals about their median; in standard
    deviations where most residuals agree.
    """
    residuals = values - answers
    year = residuals[:fitted][np.isfinite(residuals[:fitted])]
    if not len(year):
        return np.full(len(values), np.nan)
    centre = float(np.median(year))
    spread = MAD_TO_DEVIATION * float(np.median(np.abs(year - centre)))
    if spread <= 0.0:
        spread = float(year.std()) or 1.0
    far: Floats = np.abs(residuals - centre) / spread
    return far


def reduced(far: Floats) -> dict[str, Floats]:
    """The hour's largest deviation over its temperatures and their mean.

    Missing where no temperature has one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"max": np.nanmax(far, axis=1), "mean": np.nanmean(far, axis=1)}


def explained(answers: Floats, values: Floats, fitted: int) -> float:
    """The share of a temperature's variance over the fitted hours its answers explain."""
    held = np.isfinite(answers[:fitted]) & np.isfinite(values[:fitted])
    if held.sum() < 2:
        return float("nan")
    spread = float(values[:fitted][held].var())
    missed = float(((values[:fitted][held] - answers[:fitted][held]) ** 2).mean())
    return 1.0 - missed / spread if spread > 0.0 else float("nan")


def behaviour_scores(
    hourly: Hourly, others: Sequence[Hourly], fit: Fit, seed: int
) -> tuple[dict[str, Floats], Floats]:
    """An excerpt's score series by family and reduction, and each temperature's share explained."""
    temperatures = hourly.targets.shape[1]
    own = np.full(hourly.targets.shape, np.nan)
    fleet = np.full(hourly.targets.shape, np.nan)
    shares = np.full(temperatures, np.nan)
    for target in range(temperatures):
        answers = answers_of(hourly, target, fit, seed)
        shares[target] = explained(answers, hourly.targets[:, target], hourly.fitted)
        own[:, target] = deviations(answers, hourly.targets[:, target], hourly.fitted)
        if others:
            fleet[:, target] = deviations(
                fleet_answers(hourly, others, target, fit, seed),
                hourly.targets[:, target],
                hourly.fitted,
            )
    scores = {f"{FAMILY}/{name}": series for name, series in reduced(own).items()}
    if others:
        scores |= {f"{FLEET}/{name}": series for name, series in reduced(fleet).items()}
    return scores, shares


def others_of(event: Event, every: Sequence[Event]) -> list[Event]:
    """The farm's excerpts of other turbines, which a model for a turbine with no history reads.

    The turbine's own other excerpts are left out: their fitted years can hold this excerpt's
    scored rows.
    """
    return [other for other in every if other.farm == event.farm and other.turbine != event.turbine]


def laid(root: Path, event: Event, out: Path) -> Path:  # pragma: no cover - reads files
    """An excerpt's hours as the model reads them, written anew by every run."""
    path = out / f"hourly-{event.farm}-{event.event}.npz"
    excerpt = read_excerpt(root, event)
    hours = Hours.of(excerpt)
    hourly = Hourly.of(excerpt, hours, sensors_of(root, event.farm))
    # Written under another name first, so a run that is stopped leaves no half-written file.
    partial = out / f"partial-{event.farm}-{event.event}.npz"
    np.savez_compressed(
        partial,
        features=hourly.features,
        targets=hourly.targets,
        fitted=hours.fitted,
        first=hours.first,
        last=hours.last,
    )
    partial.replace(path)
    return path


def written(
    own: Path, others: Sequence[Path], out: Path, form: str, seed: int
) -> str:  # pragma: no cover - reads files
    """One excerpt's score file from its hours and the other turbines'."""
    if "torch" in sys.modules:
        raise SystemExit("torch is loaded: XGBoost's OpenMP must not share its process")
    # One thread a fit: the fits are small, and the excerpts run side by side.
    fit = xgboost_fit(1) if form == "trees" else ridge_fit
    held = np.load(own)
    hourly = Hourly(held["features"], held["targets"], int(held["fitted"]))
    beside = []
    for path in others:
        other = np.load(path)
        beside.append(Hourly(other["features"], other["targets"], int(other["fitted"])))
    scores, shares = behaviour_scores(hourly, beside, fit, seed)
    names = sorted(scores)
    name = own.name.removeprefix("hourly-")
    np.savez_compressed(
        out / f"behaviour-{name}",
        arms=np.asarray(names),
        scores=np.stack([scores[series] for series in names]).astype(np.float32),
        fitted=held["fitted"],
        first=held["first"],
        last=held["last"],
        explained=shares,
    )
    return (
        f"{name.removesuffix('.npz')}: {hourly.targets.shape[1]} temperatures, median share "
        f"explained {np.nanmedian(shares):.3f}, least {np.nanmin(shares):.3f}"
    )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--form", choices=("trees", "ridge"), required=True)
    parser.add_argument("--farm", nargs="+", default=list(FARMS), choices=FARMS)
    parser.add_argument("--event", nargs=2, action="append", default=[], metavar=("FARM", "NUMBER"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=8)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    every = [event for event in read_events(arguments.root) if event.farm in arguments.farm]
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        paths = dict(
            zip(
                ((event.farm, event.event) for event in every),
                pool.map(laid, [arguments.root] * len(every), every, [arguments.out] * len(every)),
                strict=True,
            )
        )
        read = chosen(every, arguments.farm, arguments.event)
        lines = pool.map(
            written,
            [paths[event.farm, event.event] for event in read],
            [
                [paths[other.farm, other.event] for other in others_of(event, every)]
                for event in read
            ],
            [arguments.out] * len(read),
            [arguments.form] * len(read),
            [arguments.seed] * len(read),
        )
        for line in lines:
            print(line, flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
