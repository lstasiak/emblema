r"""The channel-local departures over the fault benchmark's excerpts, hour by hour.

Each excerpt read is laid on its clock hours and scored by the package's detector of departures
(``ChannelDepartures``), every channel of an hour against the same channel over the hours before
it: the twelve detectors the telemetry's verdict reads, the largest or the mean departure over
the channels, of the hour's summary or of its six readings, over a history of 6, 24 or 72 hours.
The channels are a farm's averages but its angles and counters, each scored as a publication
scores a channel, from its median in robust deviations over the fitted year, so the floor under
a channel's spread does not hang on another channel's unit. An hour is scored where it is whole
and half its history holds a reading; nothing of the scored window enters the scale or the
floor. In the farm whose status says nothing of a scored window every hour of that window is
read, the stopped turbine's among them.

Beside the twelve, descriptively, by the hour's summary alone: the same over the component
temperatures, which a model of normal behaviour answers; the same with the scale and the floor
taken from the excerpt's first 7, 30 and 90 days, as a turbine with that much history would
have them; and the answer with no model, the share of an hour's rows not of normal operation.

    uv run scripts/care_departure_scores.py --root data/raw/care/CARE_To_Compare --out DIR \\
        [--farm A B C] [--event FARM NUMBER]... [--jobs 4]

Writes ``departures-<farm>-<event>.npz`` per excerpt: ``arms``, ``scores`` (``[arms, hours]``,
missing where an hour is not scored), the hours ``fitted``, ``first`` and ``last``, and per row
of the file its hour, whether it is fitted and, as ``row_normal``, whether it is read: of normal
operation, or any scored row of the farm whose status says nothing there.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.detection.channel_departures import (  # noqa: E402
    ChannelDepartures,
)
from emblema.evaluation.adapters.detection.channel_local_detector import (  # noqa: E402
    ChannelLocalDetector,
)
from emblema.evaluation.domain.detection.channel_history import ChannelHistory  # noqa: E402
from emblema.evaluation.domain.detection.channel_reduction import ChannelReduction  # noqa: E402
from emblema.evaluation.domain.detection.departure_cells import DepartureCells  # noqa: E402
from scripts.care_excerpts import (  # noqa: E402
    FARMS,
    Event,
    Excerpt,
    Hours,
    Sensors,
    cells_of,
    chosen,
    read_events,
    read_excerpt,
    robust_scores,
    sensors_of,
)

Floats = NDArray[np.float64]

SPANS = (6, 24, 72)
FAMILY = "channel_local"
TEMPERATURES = "channel_local_temperatures"
AVAILABILITY = "availability"
# The days of history a new turbine's scale and floor are taken from, beside the whole year.
FIRST_DAYS = (7, 30, 90)
HOURS_PER_DAY = 24
# What the families read beside the verdict's are scored by: the hour's summary alone, since the
# readings' departures over 72 hours cost two thirds of an excerpt's time.
BESIDE = (DepartureCells.WINDOW,)


def arm_of(reduction: ChannelReduction, departure: DepartureCells, span: int) -> str:
    return f"{reduction.value}-{departure.value}-{span}h"


def first_days(days: int) -> str:
    return f"{FAMILY}_first{days}d"


def departure_scores(
    readings: Floats, known: int, departures: Sequence[DepartureCells] = tuple(DepartureCells)
) -> dict[str, Floats]:
    """The detectors' scores of every hour, missing where an hour is not scored.

    Args:
        readings: ``[hours, channels, 6]``, missing where a channel is not read in an hour.
        known: How many hours, from the first, the scale and the floor are taken from.
        departures: What departs, the hour's summary or each of its readings; both make the
            twelve detectors.
    """
    scored = robust_scores(readings, known)
    hours = len(scored)
    ends = np.arange(1.0, hours + 1.0)
    every = np.arange(hours, dtype=np.int64)
    scores: dict[str, Floats] = {}
    for departure in departures:
        laid = ChannelDepartures.laid(
            ChannelHistory(span=max(SPANS), departure=departure),
            cells_of(scored),
            ends_at=ends.tolist(),
            units=["excerpt"] * hours,
            window_length=1.0,
        )
        present = laid.present(every)
        held = present.any(axis=1)
        floor = laid.floor_over(every[:known])
        for span in SPANS:
            history = ChannelHistory(span=span, departure=departure)
            read = replace(
                laid,
                history=history,
                before=ChannelDepartures.windows_before(
                    ends, length=1.0, count=history.windows_before
                ),
            )
            far = read.of(every, floor)
            loaded = ((read.before >= 0) & held[np.maximum(read.before, 0)]).sum(axis=1)
            answered = held & (loaded >= history.least_history)
            for reduction in ChannelReduction:
                reduced = ChannelLocalDetector.reduced(far, present, reduction)
                scores[arm_of(reduction, departure, span)] = np.where(answered, reduced, np.nan)
    return scores


def excerpt_scores(excerpt: Excerpt, sensors: Sensors) -> tuple[Hours, dict[str, Floats]]:
    """Every score series of one excerpt, named by family and detector."""
    hours = Hours.of(excerpt)
    scores: dict[str, Floats] = {AVAILABILITY: hours.abnormal_share(excerpt.normal)}
    channels = hours.readings(excerpt.values[:, excerpt.columns_at(sensors.channels)])
    for arm, series in departure_scores(channels, hours.fitted).items():
        scores[f"{FAMILY}/{arm}"] = series
    for days in FIRST_DAYS:
        known = min(days * HOURS_PER_DAY, hours.fitted)
        for arm, series in departure_scores(channels, known, BESIDE).items():
            scores[f"{first_days(days)}/{arm}"] = series
    temperatures = hours.readings(excerpt.values[:, excerpt.columns_at(sensors.temperatures)])
    for arm, series in departure_scores(temperatures, hours.fitted, BESIDE).items():
        scores[f"{TEMPERATURES}/{arm}"] = series
    return hours, scores


def arrays_of(excerpt: Excerpt, hours: Hours, scores: dict[str, Floats]) -> dict[str, Any]:
    """What an excerpt's file holds: its scores by detector, its hours and its rows.

    The arrays are of several types, and numpy's writer takes them as keywords of any.
    """
    names = sorted(scores)
    return {
        "arms": np.asarray(names),
        "scores": np.stack([scores[name] for name in names]).astype(np.float32),
        "fitted": np.asarray(hours.fitted),
        "first": np.asarray(hours.first),
        "last": np.asarray(hours.last),
        "row_hour": hours.of_row,
        "row_fitted": excerpt.fitted,
        # The rows a detector reads, which in the first farm's scored window is every row.
        "row_normal": excerpt.read,
        "row_id": excerpt.ids,
    }


def written(root: Path, event: Event, out: Path) -> str:  # pragma: no cover - reads files
    excerpt = read_excerpt(root, event)
    hours, scores = excerpt_scores(excerpt, sensors_of(root, event.farm))
    np.savez_compressed(
        out / f"departures-{event.farm}-{event.event}.npz", **arrays_of(excerpt, hours, scores)
    )
    answered = int((~np.isnan(scores[f"{FAMILY}/{names_of_family()[0]}"])).sum())
    return (
        f"farm {event.farm}, excerpt {event.event}: {hours.count} hours, {int(hours.whole.sum())} "
        f"whole, {answered} scored"
    )


def names_of_family() -> list[str]:
    """The twelve detectors of the family, in a fixed order."""
    return [
        arm_of(reduction, departure, span)
        for reduction in ChannelReduction
        for departure in DepartureCells
        for span in SPANS
    ]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads files
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--farm", nargs="+", default=list(FARMS), choices=FARMS)
    parser.add_argument("--event", nargs=2, action="append", default=[], metavar=("FARM", "NUMBER"))
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    events = chosen(read_events(arguments.root), arguments.farm, arguments.event)
    with ProcessPoolExecutor(max_workers=arguments.jobs) as pool:
        lines = pool.map(
            written, [arguments.root] * len(events), events, [arguments.out] * len(events)
        )
        for line in lines:
            print(line, flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
