"""Draw how a window of sensor data becomes the input of the encoder, from the tokens stored.

One row per window. On the left, what the sensors recorded: each sensor on its own lane, on its
own scale and at its own times. On the right, what the model reads: one row per reading, each
naming its sensor, the value after that sensor's own scaling, its position in the window and the
time since that sensor's previous reading; static facts are rows with no time. The files are the
ones ``token_view_report.py`` wrote; nothing is tokenised here.

    uv run scripts/token_view_figures.py data/report/token-view
    uv run scripts/token_view_figures.py <dir> --figure docs/images/token-view.png
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# A file is written, nothing is shown: the backend must not need a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

FIGURE = REPO_ROOT / "docs" / "images" / "token-view.png"
TITLES = {
    "physionet2012": "Intensive-care stay (PhysioNet 2012): each sensor on its own clock",
    "cmapss": "Turbofan engine (NASA C-MAPSS): every sensor every cycle",
}
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0", "#d64f8c", "#4a4a4a")
# The token written out in full on each row: one that is easy to find and not at an edge.
# How many timed tokens the table lists for a window.
LISTED = 8


@dataclass(frozen=True)
class Row:
    """One token as stored; a plain record read from the report's file."""

    channel: str
    shown: bool
    timeless: bool
    raw_time: float | None
    raw_value: float
    value: float
    time: float
    gap: float


@dataclass(frozen=True)
class Window:
    """One window as stored: where it was cut and how much it holds."""

    corpus: str
    unit: str
    start: float
    end: float
    time_unit: str
    tokens: int
    channels: int


def read(directory: Path) -> tuple[list[Window], dict[str, list[Row]]]:
    with (directory / "windows.csv").open() as file:
        windows = [
            Window(
                r["corpus"],
                r["unit"],
                float(r["start"]),
                float(r["end"]),
                r["time_unit"],
                int(r["tokens"]),
                int(r["channels"]),
            )
            for r in csv.DictReader(file)
        ]
    tokens: dict[str, list[Row]] = defaultdict(list)
    with (directory / "tokens.csv").open() as file:
        for r in csv.DictReader(file):
            tokens[r["corpus"]].append(
                Row(
                    channel=r["channel"],
                    shown=r["shown"] == "True",
                    timeless=r["timeless"] == "True",
                    raw_time=float(r["raw_time"]) if r["raw_time"] else None,
                    raw_value=float(r["raw_value"]),
                    value=float(r["value"]),
                    time=float(r["time"]),
                    gap=float(r["gap"]),
                )
            )
    return windows, tokens


def colours(rows: list[Row]) -> dict[str, str]:
    """A colour per shown sensor, in the order the sensors first appear, static ones last."""
    order = [r.channel for r in rows if r.shown and not r.timeless]
    order += [r.channel for r in rows if r.shown and r.timeless]
    names = list(dict.fromkeys(order))
    return dict(zip(names, PALETTE, strict=False))


def draw_recorded(axes: Axes, window: Window, rows: list[Row], hue: dict[str, str]) -> None:
    """Each shown sensor on a lane of its own, its readings scaled into the lane."""
    timed = [r for r in rows if r.shown and not r.timeless]
    lanes = list(dict.fromkeys(r.channel for r in timed))
    for lane, channel in enumerate(reversed(lanes)):
        points = sorted((r.raw_time or 0.0, r.raw_value) for r in timed if r.channel == channel)
        values = [v for _, v in points]
        low, high = min(values), max(values)
        spread = high - low or 1.0
        ys = [lane + 0.7 * ((v - low) / spread - 0.5) for v in values]
        xs = [t for t, _ in points]
        axes.plot(xs, ys, color=hue[channel], linewidth=0.8, alpha=0.6)
        axes.scatter(xs, ys, color=hue[channel], s=9, zorder=3)
        axes.text(
            window.start - 0.02 * (window.end - window.start),
            lane,
            f"{channel}\n{len(points)} readings",
            ha="right",
            va="center",
            fontsize=8,
            color=hue[channel],
        )
    statics = [r for r in rows if r.shown and r.timeless]
    if statics:
        facts = ", ".join(static_label(r) for r in statics)
        axes.text(
            0.0,
            -0.3,
            f"recorded once, with no time: {facts}",
            transform=axes.transAxes,
            fontsize=8,
            color="#4a4a4a",
        )
    axes.set_xlim(window.start, window.end)
    axes.set_ylim(-0.6, len(lanes) - 0.4)
    axes.set_yticks([])
    axes.set_xlabel(window.time_unit, fontsize=8)
    axes.tick_params(labelsize=7)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)


def static_label(row: Row) -> str:
    """A static fact the way a reader would say it: a ward is named, a number is shown."""
    if "/" in row.channel:
        group, member = row.channel.split("/", 1)
        return f"{group} = {member.replace('_', ' ')}"
    return f"{row.channel} = {row.raw_value:g}"


def listed(rows: list[Row], count: int) -> list[Row]:
    """The tokens the table lists: the static ones, then a run of consecutive timed ones.

    The run is the stretch of ``count`` timed tokens of the shown sensors where the most sensors
    take turns, so that the gaps a sensor's readings are apart can be read off beside each other.
    """
    statics = [r for r in rows if r.shown and r.timeless]
    timed = sorted((r for r in rows if r.shown and not r.timeless), key=lambda r: r.time)
    if len(timed) <= count:
        return statics + timed
    starts = range(len(timed) - count + 1)
    best = max(starts, key=lambda i: (len({r.channel for r in timed[i : i + count]}), -i))
    return statics + timed[best : best + count]


def draw_tokens(axes: Axes, window: Window, rows: list[Row], hue: dict[str, str]) -> None:
    """What the encoder is given: one row per reading, whatever the sensor, in no fixed layout."""
    axes.axis("off")
    columns = (
        (0.0, "sensor", "left"),
        (0.36, "reading", "right"),
        (0.56, "value", "right"),
        (0.75, "time", "right"),
        (0.94, "gap", "right"),
    )
    top, step = 0.94, 0.075
    for x, name, align in columns:
        axes.text(x, top, name, fontsize=8.5, weight="bold", ha=align, family="monospace")
    shown = listed(rows, LISTED)
    for number, token in enumerate(shown, start=1):
        y = top - number * step
        # A ward is a presence token: its reading is the ward's name, not the 1 that marks it.
        sensor, _, member = token.channel.partition("/")
        cells = (
            sensor.removesuffix("Type"),
            member.replace("_", " ") if member else f"{token.raw_value:g}",
            f"{token.value:+.2f}",
            "—" if token.timeless else f"{token.time:.3f}",
            "—" if token.timeless else f"{token.gap:.3f}",
        )
        for (x, _, align), cell in zip(columns, cells, strict=True):
            axes.text(
                x,
                y,
                cell,
                fontsize=8.5,
                ha=align,
                family="monospace",
                color=hue.get(token.channel, "#4a4a4a") if x == 0.0 else "#222222",
            )
    y = top - (len(shown) + 1) * step
    axes.text(0.0, y, "⋮", fontsize=10, color="#4a4a4a")
    axes.text(
        0.0,
        y - step * 1.2,
        f"{window.tokens:,} rows in all, from {window.channels} sensors",
        fontsize=8.5,
        color="#4a4a4a",
    )
    axes.text(
        0.0,
        y - step * 2.4,
        "value: scaled per sensor · time: position in the window, 0 to 1\n"
        "gap: time since that sensor's previous reading, as a share of the window",
        va="top",
        fontsize=7.5,
        color="#4a4a4a",
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)


def draw(directory: Path, figure: Path) -> Path:
    windows, tokens = read(directory)
    fig = plt.figure(figsize=(11, 3.4 * len(windows)))
    grid = fig.add_gridspec(len(windows), 2, width_ratios=(1.15, 1.0), wspace=0.3, hspace=0.75)
    for row, window in enumerate(windows):
        rows = tokens[window.corpus]
        hue = colours(rows)
        recorded = fig.add_subplot(grid[row, 0])
        draw_recorded(recorded, window, rows, hue)
        recorded.set_title(
            TITLES.get(window.corpus, window.corpus), fontsize=10, loc="left", x=-0.2, pad=18
        )
        draw_tokens(fig.add_subplot(grid[row, 1]), window, rows, hue)
    fig.text(0.08, 0.99, "What the sensors record", fontsize=11, weight="bold", va="top")
    fig.text(0.56, 0.99, "What the model reads", fontsize=11, weight="bold", va="top")
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    arguments = parser.parse_args()
    print(draw(arguments.directory, arguments.figure))


if __name__ == "__main__":
    main()
