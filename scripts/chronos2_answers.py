r"""A published zero-shot forecaster's answer for each cell's hidden hour, from its hourly means.

Chronos-2 is asked, for every cell the pretext's reading scored, for the hour after the 47 hourly
means the context file holds, and its median is kept as the answer for the hour's mean. The
means are in the corpus's normalised units, each channel scaled by the statistics of its
training months, as the published comparison on this telemetry scaled them; the model then
scales each series on its own inside, as it does for every input. The channels of one window
are handed over together, so the model reads them as the variates of one series, as the backbone
reads a window; ``--mode univariate`` hands each channel alone. A cell with no finite mean in
its context is refused: the reading never scores one.

The forecaster is not a dependency of this package, so this runs in its own environment:

    uv run --isolated --no-project --python 3.12 --with chronos-forecasting \\
        scripts/chronos2_answers.py --context DIR/context-<label>.csv \\
        --out DIR/chronos2-<label>.csv [--mode multivariate] [--model amazon/chronos-2] \\
        [--device mps] [--limit N]

Writes one row per cell: window, axis, channel and the forecast of the hour's mean.
"""

import argparse
import csv
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

COLUMNS = ("window", "axis", "channel", "mean_chronos2")
MODES = ("multivariate", "univariate")

Predict = Callable[[Sequence[NDArray[np.float32]]], Sequence[NDArray[np.float32]]]


@dataclass(frozen=True)
class ContextRow:
    """One scored cell's hours before the hidden one, NaN where the channel read nothing."""

    window: int
    axis: str
    channel: int
    means: NDArray[np.float32]


def read_context(path: Path) -> list[ContextRow]:
    """The context file's rows, in order.

    Raises:
        ValueError: If a row holds no finite mean, which no scored cell does.
    """
    rows = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        hours = [k for k, name in enumerate(header) if name.startswith("h")]
        for line in reader:
            means = np.array(
                [float(line[k]) if line[k] else np.nan for k in hours], dtype=np.float32
            )
            if not np.isfinite(means).any():
                raise ValueError(
                    f"window {line[0]} channel {line[2]} has no finite mean in its context"
                )
            rows.append(ContextRow(int(line[0]), line[1], int(line[2]), means))
    return rows


def grouped(rows: Sequence[ContextRow], mode: str) -> list[list[ContextRow]]:
    """The cells handed to the model together: a window's channels, or each cell alone.

    Raises:
        ValueError: If the mode is not one of ``MODES``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if mode == "univariate":
        return [[row] for row in rows]
    groups: dict[tuple[int, str], list[ContextRow]] = {}
    for row in rows:
        groups.setdefault((row.window, row.axis), []).append(row)
    return list(groups.values())


def median_index(quantiles: Sequence[float]) -> int:
    """Which of the model's quantile levels is the median.

    Raises:
        ValueError: If the model gives no median.
    """
    for k, level in enumerate(quantiles):
        if abs(level - 0.5) < 1e-9:
            return k
    raise ValueError(f"the model's quantiles hold no median: {list(quantiles)}")


def forecast(
    groups: Sequence[Sequence[ContextRow]],
    predict: Predict,
    quantiles: Sequence[float],
    *,
    batch: int = 200,
    say: Callable[[str], None] = lambda _: None,
) -> list[tuple[ContextRow, float]]:
    """Each cell with the model's median for its hidden hour, the groups handed in batches.

    ``predict`` takes 2-d arrays ``[variates, hours]`` and returns, for each, ``[variates,
    quantiles, 1]``.

    Raises:
        ValueError: If the model answers another number of variates than it was handed.
    """
    at = median_index(quantiles)
    answered: list[tuple[ContextRow, float]] = []
    started = time.monotonic()
    for start in range(0, len(groups), batch):
        handed = groups[start : start + batch]
        outputs = predict([np.stack([row.means for row in group]) for group in handed])
        for group, output in zip(handed, outputs, strict=True):
            if output.shape[0] != len(group):
                raise ValueError(
                    f"the model answered {output.shape[0]} variates for {len(group)} handed"
                )
            answered.extend((row, float(output[k, at, 0])) for k, row in enumerate(group))
        say(f"{len(answered)} cells in {time.monotonic() - started:.0f} s")
    return answered


def write_forecasts(path: Path, answered: Sequence[tuple[ContextRow, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row, mean in answered:
            writer.writerow((row.window, row.axis, row.channel, f"{mean:.6g}"))


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="multivariate")
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=200, help="groups handed at once")
    parser.add_argument("--limit", type=int, help="read the first N windows only, for a trial")
    arguments = parser.parse_args(argv)

    import chronos  # type: ignore[import-not-found,unused-ignore]  # ty: ignore[unresolved-import]
    import torch  # type: ignore[import-not-found,unused-ignore]

    rows = read_context(arguments.context)
    if arguments.limit is not None:
        kept = sorted({row.window for row in rows})[: arguments.limit]
        rows = [row for row in rows if row.window in kept]
    groups = grouped(rows, arguments.mode)
    started = time.monotonic()
    pipeline = chronos.Chronos2Pipeline.from_pretrained(
        arguments.model, device_map=arguments.device, dtype=torch.float32
    )
    print(
        f"{arguments.model} loaded in {time.monotonic() - started:.0f} s, "
        f"{sum(p.numel() for p in pipeline.model.parameters()) / 1e6:.0f}M parameters"
    )

    def predict(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        return [t.cpu().numpy() for t in pipeline.predict(list(batch), prediction_length=1)]

    answered = forecast(
        groups,
        predict,
        pipeline.quantiles,
        batch=arguments.batch,
        say=lambda text: print(text, file=sys.stderr),
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write_forecasts(arguments.out, answered)
    print(
        f"{len(answered)} cells of {len(groups)} groups ({arguments.mode}) answered in "
        f"{time.monotonic() - started:.0f} s"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
