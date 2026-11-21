r"""A published forecaster's answer for each cell of a turbine read as a new system.

Chronos-2 forecasts the hour of each cell ``fleet_cold_start_cells.py`` wrote, the new turbine's
hours before its start unknown. Unless said otherwise, the window's channels — the seventeen
temperatures and the four signals of context — are handed together as the variates of one
series, as the published forecaster was read on the satellites. The rungs differ in what it is
handed:

- ``chronos2_47``: the 47 hours before;
- ``chronos2_47_each``: the same hours, each scored channel alone;
- ``chronos2_own``: every hour since the start, up to the most the model reads;
- ``chronos2_neighbours``: the 47 hours, and as covariates over them each temperature's mean
  over the farm's other turbines, which share the new one's weather and whose hours are known;
- ``chronos2_neighbours_concurrent``, the ceiling of that path and no forecaster: the same, with
  the other turbines' mean in the forecast hour itself handed as known;
- ``chronos2_nbm``, of normal behaviour: the seventeen temperatures' 47 hours, and as covariates
  the six signals of the operating conditions — the wind, the power, the rotor's speed, a
  blade's angle and the two ambient temperatures — over the same hours and in the forecast hour
  itself, known as a turbine measures them; only where all six read in that hour;
- ``chronos2_nbm_currents``, beside: the same with the pitch motors' currents too;
- ``chronos2_nbm_own``, beside: of normal behaviour on every hour since the start, up to the
  most the model reads, as ``chronos2_own`` is on the window's channels.

The forecaster is not a dependency of this package, so this runs in its own environment, and
reads nothing of it:

    uv run --isolated --no-project --python 3.12 --with chronos-forecasting==2.3.2 python -I \\
        scripts/fleet_chronos2_answers.py --hourly DIR --cells DIR/answers-<label>.csv \\
        --out DIR/chronos2-<label>.csv [--device mps]

Writes one row per cell: window, axis, channel and each rung's forecast of the hour's mean.
"""

import argparse
import csv
import faulthandler
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run in the forecaster's own environment: only the sibling scripts are reached from the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_answers import median_index  # noqa: E402
from scripts.chronos2_ladder import CONTEXT, before, read_cells  # noqa: E402
from scripts.fleet_hours import (  # noqa: E402
    CONDITIONS,
    CURRENTS,
    TARGETS,
    WINDOW,
    Turbine,
    hour_of,
    loaded,
    parsed_axis,
)

Floats = NDArray[np.float64]
Key = tuple[int, str, int]
EACH = "chronos2_47_each"
OWN = "chronos2_own"
NEIGHBOURS = "chronos2_neighbours"
CONCURRENT = "chronos2_neighbours_concurrent"
NBM = "chronos2_nbm"
NBM_CURRENTS = "chronos2_nbm_currents"
NBM_OWN = "chronos2_nbm_own"
# Each normal-behaviour rung's covariates, read before and in the forecast hour.
KNOWN = {NBM: CONDITIONS, NBM_CURRENTS: (*CONDITIONS, *CURRENTS), NBM_OWN: CONDITIONS}
RUNGS = ("chronos2_47", EACH, OWN, NEIGHBOURS, CONCURRENT, NBM, NBM_CURRENTS, NBM_OWN)
COLUMNS = ("window", "axis", "channel", *(f"mean_{rung}" for rung in RUNGS))
STARTUP_SECONDS = 300
# Chronos-2's inputs: an array of variates, or a dict of target and covariates.
Input = Any
Predict = Callable[[Sequence[Input]], list[NDArray[np.float32]]]


def neighbours_mean(turbines: Sequence[Turbine], new: str) -> Floats:
    """``[hours, temperatures]``: each temperature's mean over the other turbines that read it."""
    others = np.stack([t.values[:, : len(TARGETS)] for t in turbines if t.name != new])
    read = np.isfinite(others)
    counts = read.sum(axis=0)
    sums = np.where(read, others, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: Floats = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def covariates(neighbours: Floats, hours: slice) -> dict[str, Floats]:
    return {f"n{c}": neighbours[hours, c] for c in range(len(TARGETS))}


def nbm_input(rung: str, window: int, start: int, values: Floats, cap: int) -> Input | None:
    """What a normal-behaviour rung is handed for one window, or None without its hour.

    The temperatures are the target; the rung's signals are covariates over the same hours —
    the 47 before, or every one since ``start`` up to ``cap`` — and, known, in the forecast
    hour. ``values`` are the new turbine's, every channel read, unknown before ``start``.
    """
    known = KNOWN[rung]
    if not np.isfinite(values[window, list(known)]).all():
        return None
    length = min(window - start, cap) if rung == NBM_OWN else CONTEXT
    return {
        "target": before(values[:, : len(TARGETS)], window, length),
        "past_covariates": {f"s{i}": before(values[:, [i]], window, length)[0] for i in known},
        "future_covariates": {f"s{i}": values[window : window + 1, i] for i in known},
    }


def rung_input(
    rung: str, window: int, start: int, values: Floats, neighbours: Floats, cap: int
) -> Input:
    """What the forecaster is handed for one window on a rung of the window's channels.

    ``values`` are the new turbine's, every channel read, unknown before ``start``.
    """
    values = values[:, : len(WINDOW)]
    if rung == OWN:
        return before(values, window, min(window - start, cap))
    target = before(values, window, CONTEXT)
    if rung == "chronos2_47":
        return target
    handed: dict[str, Any] = {
        "target": target,
        "past_covariates": covariates(neighbours, slice(window - CONTEXT, window)),
    }
    if rung == CONCURRENT:
        handed["future_covariates"] = covariates(neighbours, slice(window, window + 1))
    return handed


def answered(
    cells: Sequence[Key],
    turbines: Sequence[Turbine],
    predict: Predict,
    median: int,
    *,
    cap: int,
    batch: int = 32,
    say: Callable[[str], None] = lambda _: None,
) -> dict[str, dict[Key, float]]:
    """Every rung's median for every cell.

    Raises:
        ValueError: If a cell's axis names a turbine not loaded, the forecaster answers another
            number of variates, or a cell no finite value.
    """
    by_name = {t.name: t for t in turbines}
    windows = sorted({(w, a) for w, a, _ in cells}, key=lambda k: (k[1], k[0]))
    scored = set(cells)
    news: dict[str, tuple[int, Floats]] = {}
    means: dict[str, Floats] = {}
    for _, axis in windows:
        if axis in news:
            continue
        name, day = parsed_axis(axis)
        if name not in by_name:
            raise ValueError(f"{axis} names a turbine not loaded")
        start = hour_of(day)
        news[axis] = (start, by_name[name].before(start).values)
        if name not in means:
            means[name] = neighbours_mean(turbines, name)
    out: dict[str, dict[Key, float]] = {rung: {} for rung in RUNGS}
    started = time.monotonic()
    for rung in RUNGS:
        if rung == EACH:
            out[rung] = alone(cells, news, predict, median, batch=batch * len(WINDOW))
        else:
            handed_all = (
                (
                    (window, axis),
                    nbm_input(rung, window, news[axis][0], news[axis][1], cap)
                    if rung in KNOWN
                    else rung_input(
                        rung,
                        window,
                        news[axis][0],
                        news[axis][1],
                        means[parsed_axis(axis)[0]],
                        cap,
                    ),
                )
                for window, axis in windows
            )
            items = [(key, handed) for key, handed in handed_all if handed is not None]
            variates = len(TARGETS) if rung in KNOWN else len(WINDOW)
            for first in range(0, len(items), batch):
                part = items[first : first + batch]
                outputs = predict([handed for _, handed in part])
                for ((window, axis), _), output in zip(part, outputs, strict=True):
                    if output.shape[0] != variates:
                        raise ValueError(
                            f"the model answered {output.shape[0]} variates for {variates}"
                        )
                    for channel in range(len(TARGETS)):
                        key = (window, axis, channel)
                        if key not in scored:
                            continue
                        mean = float(output[channel, median, 0])
                        if not np.isfinite(mean):
                            raise ValueError(f"{rung} answered no finite value for {key}")
                        out[rung][key] = mean
        say(f"{rung}: {len(out[rung])} cells in {time.monotonic() - started:.0f} s")
    return out


def alone(
    cells: Sequence[Key],
    news: Mapping[str, tuple[int, Floats]],
    predict: Predict,
    median: int,
    *,
    batch: int,
) -> dict[Key, float]:
    """Each cell's median with its channel's 47 hours handed alone.

    Raises:
        ValueError: If the forecaster answers more than one variate, or no finite value.
    """
    out: dict[Key, float] = {}
    for first in range(0, len(cells), batch):
        part = cells[first : first + batch]
        outputs = predict([before(news[a][1], w, CONTEXT)[c][None, :] for w, a, c in part])
        for key, output in zip(part, outputs, strict=True):
            if output.shape[0] != 1:
                raise ValueError(f"the model answered {output.shape[0]} variates for one")
            mean = float(output[0, median, 0])
            if not np.isfinite(mean):
                raise ValueError(f"{EACH} answered no finite value for {key}")
            out[key] = mean
    return out


def write(path: Path, cells: Sequence[Key], out: Mapping[str, Mapping[Key, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for key in cells:
            shown = (f"{out[r][key]:.6g}" if key in out[r] else "" for r in RUNGS)
            writer.writerow((*key, *shown))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hourly", type=Path, required=True, help="the farm's hourly means")
    parser.add_argument("--cells", type=Path, required=True, help="fleet_cold_start_cells's")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=32, help="windows handed at once")
    arguments = parser.parse_args(argv)

    # A start has sometimes waited on a lock for half an hour before the model loaded; past
    # this every thread's stack is printed and the process stops, so the caller can run it again.
    faulthandler.dump_traceback_later(STARTUP_SECONDS, exit=True)
    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    names = sorted(p.stem.removeprefix("hourly-") for p in arguments.hourly.glob("hourly-*.npz"))
    turbines = loaded(arguments.hourly, names)
    cells = read_cells(arguments.cells)
    started = time.monotonic()
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        arguments.model, device_map=arguments.device, dtype=torch.float32
    )
    faulthandler.cancel_dump_traceback_later()
    cap = int(pipeline.model_context_length)

    def predict(batch: Sequence[Input]) -> list[NDArray[np.float32]]:
        return [
            t.cpu().numpy()
            for t in pipeline.predict(list(batch), prediction_length=1, context_length=cap)
        ]

    out = answered(
        cells,
        turbines,
        predict,
        median_index(pipeline.quantiles),
        cap=cap,
        batch=arguments.batch,
        say=lambda text: print(text, file=sys.stderr, flush=True),
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.out, cells, out)
    print(f"{len(cells)} cells answered in {time.monotonic() - started:.0f} s, context up to {cap}")


if __name__ == "__main__":  # pragma: no cover
    main()
