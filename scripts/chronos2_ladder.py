r"""A published forecaster's answer for each cold-start cell, on rungs of context.

Chronos-2 forecasts the hour after a drawn hour of a mission read as a new system. Unless said
otherwise, every channel of the mission is handed together as the variates of one series, as the
published forecaster was read on 2026-11-19. The rungs differ in what it is handed:

- ``chronos2_47``: the 47 hours before, the context of that reading;
- ``chronos2_47_each``: the same hours, each channel handed alone;
- ``chronos2_own``: every hour since the mission's first up to the forecast hour, up to the
  most the model reads: what a zero-shot forecaster has as the hours arrive, more than the
  ridges, fitted once on day N, learn from;
- ``chronos2_unrelated``, ``chronos2_kind``, ``chronos2_analog``: the 47 hours before, and for
  each scored channel with a counterpart on the other mission, the library series
  ``cold_start_cells.py`` chose for that rung, as a covariate over the same 47 hours, the hour it
  meets standing for the forecast hour and handed as known: the 47 hours it was chosen on cover
  the target's whole context, so the rungs differ from ``chronos2_47`` in the library alone;
- ``chronos2_oracle``, the ceiling of that path and no forecaster: on the same cells, the target
  channel's own 47 hours and the forecast hour itself, handed as the library series is.

Only the cells with a counterpart are answered on the last four rungs. The forecaster is not a
dependency of this package, so this runs in its own environment, and reads nothing of it:

    uv run --isolated --no-project --python 3.12 --with chronos-forecasting==2.3.2 python -I \\
        scripts/chronos2_ladder.py --hourly DIR --mission ESA-Mission2 --library ESA-Mission1 \\
        --cells DIR/answers-<label>.csv --selections DIR/selections-<label>.csv \\
        --out DIR/chronos2-<label>.csv [--device mps]

Writes one row per cell: window, axis, channel and each rung's forecast of the hour's mean.
"""

import argparse
import csv
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run in the forecaster's own environment: only the sibling scripts are reached from the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.chronos2_answers import median_index  # noqa: E402

Floats = NDArray[np.float64]
Key = tuple[int, str, int]
CONTEXT = 47
EACH = "chronos2_47_each"
OWN = ("chronos2_47", EACH, "chronos2_own")
LADDER = ("unrelated", "kind", "analog")
ORACLE = "chronos2_oracle"
RUNGS = (*OWN, *(f"chronos2_{rung}" for rung in LADDER), ORACLE)
# Chronos-2's inputs: an array of variates, or a dict of target and covariates.
Input = Any
Predict = Callable[[Sequence[Input]], list[NDArray[np.float32]]]


def dense(hours: NDArray[np.int64], means: Floats) -> Floats:
    """``[hours, channels]`` from the first hour, NaN where nothing was read or loaded."""
    values = np.full((int(hours.max()) + 1, means.shape[1]), np.nan)
    values[hours] = means
    return values


def before(values: Floats, hour: int, length: int) -> Floats:
    """``[channels, length]``: every channel's hours before ``hour``, NaN before the first."""
    start = hour - length
    seen = values[max(start, 0) : hour].T
    return np.concatenate([np.full((values.shape[1], max(-start, 0)), np.nan), seen], axis=1)


def laid(series: Floats, meets: int) -> tuple[Floats, Floats]:
    """A library series' 47 hours before ``meets``, laid against the target's, and that hour.

    Raises:
        ValueError: If fewer than 47 hours come before ``meets``.
    """
    if meets < CONTEXT:
        raise ValueError(f"{meets} hours come before the hour met, not {CONTEXT}")
    return series[meets - CONTEXT : meets], series[meets : meets + 1]


@dataclass(frozen=True)
class Selection:
    library_channel: int
    meets: int


def read_cells(path: Path) -> list[Key]:
    with path.open(newline="") as handle:
        return [
            (int(row["window"]), row["axis"], int(row["channel"])) for row in csv.DictReader(handle)
        ]


def read_selections(path: Path) -> dict[str, dict[Key, Selection]]:
    """Each rung's library series per cell."""
    out: dict[str, dict[Key, Selection]] = {rung: {} for rung in LADDER}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["window"]), row["axis"], int(row["channel"]))
            out[row["rung"]][key] = Selection(int(row["library_channel"]), int(row["meets"]))
    return out


def rung_inputs(
    rung: str,
    window: int,
    axis: str,
    values: Floats,
    library: Floats,
    selections: Mapping[str, Mapping[Key, Selection]],
    cap: int,
) -> tuple[Input, tuple[int, ...]]:
    """What the forecaster is handed for one window on one rung, and the channels it covers.

    ``chronos2_47_each`` is handed a window's channels one by one, so ``answered`` asks for it.
    """
    if rung == "chronos2_own":
        return before(values, window, min(window, cap)), ()
    target = before(values, window, CONTEXT)
    if rung == "chronos2_47":
        return target, ()
    chosen = selections["analog" if rung == ORACLE else rung.removeprefix("chronos2_")]
    covered = tuple(sorted(c for (w, a, c) in chosen if w == window and a == axis))
    past: dict[str, Floats] = {}
    future: dict[str, Floats] = {}
    for channel in covered:
        if rung == ORACLE:
            series, meets = values[:, channel], window
        else:
            selection = chosen[(window, axis, channel)]
            series, meets = library[:, selection.library_channel], selection.meets
        past[f"c{channel}"], future[f"c{channel}"] = laid(series, meets)
    return {"target": target, "past_covariates": past, "future_covariates": future}, covered


def answered(
    cells: Sequence[Key],
    values: Floats,
    library: Floats,
    selections: Mapping[str, Mapping[Key, Selection]],
    predict: Predict,
    median: int,
    *,
    cap: int,
    batch: int = 32,
    say: Callable[[str], None] = lambda _: None,
) -> dict[str, dict[Key, float]]:
    """Every rung's median for every cell it covers.

    Windows whose covariates are the same channels are handed together, as the forecaster asks
    of one call; the channels of a window are its variates, but for ``chronos2_47_each``.

    Raises:
        ValueError: If the forecaster answers another number of variates, or a covered cell no
            finite value.
    """
    windows = sorted({(w, a) for w, a, _ in cells})
    scored = set(cells)
    out: dict[str, dict[Key, float]] = {rung: {} for rung in RUNGS}
    started = time.monotonic()
    for rung in RUNGS:
        if rung == EACH:
            out[rung] = alone(cells, values, predict, median, batch=batch * values.shape[1])
            say(f"{rung}: {len(out[rung])} cells in {time.monotonic() - started:.0f} s")
            continue
        by_schema: dict[tuple[int, ...], list[tuple[tuple[int, str], Input]]] = {}
        for window, axis in windows:
            handed, covered = rung_inputs(rung, window, axis, values, library, selections, cap)
            if rung not in OWN and not covered:
                continue
            by_schema.setdefault(covered, []).append(((window, axis), handed))
        for covered, items in by_schema.items():
            for start in range(0, len(items), batch):
                part = items[start : start + batch]
                outputs = predict([handed for _, handed in part])
                for ((window, axis), _), output in zip(part, outputs, strict=True):
                    if output.shape[0] != values.shape[1]:
                        raise ValueError(
                            f"the model answered {output.shape[0]} variates for {values.shape[1]}"
                        )
                    wanted = covered if rung not in OWN else range(values.shape[1])
                    for channel in wanted:
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
    cells: Sequence[Key], values: Floats, predict: Predict, median: int, *, batch: int
) -> dict[Key, float]:
    """Each cell's median with its channel's 47 hours handed alone.

    Raises:
        ValueError: If the forecaster answers more than one variate, or no finite value.
    """
    out: dict[Key, float] = {}
    for start in range(0, len(cells), batch):
        part = cells[start : start + batch]
        outputs = predict([before(values, w, CONTEXT)[c][None, :] for w, _, c in part])
        for key, output in zip(part, outputs, strict=True):
            if output.shape[0] != 1:
                raise ValueError(f"the model answered {output.shape[0]} variates for one")
            mean = float(output[0, median, 0])
            if not np.isfinite(mean):
                raise ValueError(f"{EACH} answered no finite value for {key}")
            out[key] = mean
    return out


def write(
    path: Path,
    cells: Sequence[Key],
    out: Mapping[str, Mapping[Key, float]],
    rungs: Sequence[str],
) -> None:
    """One row per cell: its key and each of ``rungs``' answers, blank where it gave none."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("window", "axis", "channel", *(f"mean_{rung}" for rung in rungs)))
        for key in cells:
            shown = (f"{out[r][key]:.6g}" if key in out[r] else "" for r in rungs)
            writer.writerow((*key, *shown))


def load(directory: Path, mission: str) -> Floats:
    with np.load(directory / f"hourly-{mission}.npz") as arrays:
        return dense(arrays["hours"], arrays["means"].astype(np.float64))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--cells", type=Path, required=True, help="cold_start_cells's answers")
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=32, help="windows handed at once")
    arguments = parser.parse_args(argv)

    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    values = load(arguments.hourly, arguments.mission)
    library = load(arguments.hourly, arguments.library)
    cells = read_cells(arguments.cells)
    selections = read_selections(arguments.selections)
    started = time.monotonic()
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        arguments.model, device_map=arguments.device, dtype=torch.float32
    )

    cap = int(pipeline.model_context_length)

    def predict(batch: Sequence[Input]) -> list[NDArray[np.float32]]:
        return [
            t.cpu().numpy()
            for t in pipeline.predict(list(batch), prediction_length=1, context_length=cap)
        ]

    out = answered(
        cells,
        values,
        library,
        selections,
        predict,
        median_index(pipeline.quantiles),
        cap=cap,
        batch=arguments.batch,
        say=lambda text: print(text, file=sys.stderr, flush=True),
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write(arguments.out, cells, out, RUNGS)
    print(f"{len(cells)} cells answered in {time.monotonic() - started:.0f} s, context up to {cap}")


if __name__ == "__main__":  # pragma: no cover
    main()
