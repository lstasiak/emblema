"""A wind farm's turbines as the cold-start readings take them: channels, time axis, axes.

The farm's publication holds sixty signals per turbine; a window of a cold-start reading holds
the seventeen component temperatures it scores and four signals it reads as context, and the
normal-behaviour answers also read the operating point, and in a variant the pitch motors'
currents, in the forecast hour. Hours run
from midnight UTC of 2016-01-01, the publication's origin, to the end of 2022, the last year it
holds. Only numpy and the standard library are imported, so the published forecaster's own
environment reads the turbines as the cells' script does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]

TARGETS = (
    "CPU temperature",
    "Front bearing temperature",
    "Gear oil inlet temperature",
    "Gear oil temperature",
    "Generator bearing front temperature",
    "Generator bearing rear temperature",
    "Hub temperature",
    "Nacelle temperature",
    "Rear bearing temperature",
    "Rotor bearing temp",
    "Stator temperature 1",
    "Temp. top box",
    "Temperature motor axis 1",
    "Temperature motor axis 2",
    "Temperature motor axis 3",
    "Transformer cell temperature",
    "Transformer temperature",
)
CONTEXT_SIGNALS = (
    "Wind speed",
    "Power",
    "Nacelle ambient temperature",
    "Ambient temperature (converter)",
)
WINDOW = (*TARGETS, *CONTEXT_SIGNALS)
# The operating point read in the forecast hour by the normal-behaviour answers, beside the
# window's context: signals the controller sets from the wind, not from a temperature.
OPERATING_POINT = ("Rotor speed", "Blade angle (pitch position) A")
# The pitch motors' currents, each a source of heat in a scored temperature, read in a
# descriptive variant only.
PITCH_CURRENTS = ("Motor current axis 1", "Motor current axis 2", "Motor current axis 3")
CHANNELS = (*WINDOW, *OPERATING_POINT, *PITCH_CURRENTS)
CONDITIONS = tuple(
    CHANNELS.index(c)
    for c in (
        "Wind speed",
        "Power",
        "Rotor speed",
        "Blade angle (pitch position) A",
        "Nacelle ambient temperature",
        "Ambient temperature (converter)",
    )
)
CURRENTS = tuple(CHANNELS.index(c) for c in PITCH_CURRENTS)
POWER = CHANNELS.index("Power")
# The publication's time axis starts here and holds no hour from the year it ends.
ORIGIN = date(2016, 1, 1)
READ_UNTIL = date(2023, 1, 1)
HOURS_PER_DAY = 24


def hour_of(day: date) -> int:
    return (day - ORIGIN).days * HOURS_PER_DAY


def axis_of(turbine: str, start: date) -> str:
    return f"{turbine}@{start.isoformat()}"


def parsed_axis(axis: str) -> tuple[str, date]:
    """The turbine and start an axis names.

    Raises:
        ValueError: If the axis is not ``<turbine>@<date>``.
    """
    turbine, _, day = axis.partition("@")
    if not turbine or not day:
        raise ValueError(f"{axis!r} names no turbine and start")
    return turbine, date.fromisoformat(day)


@dataclass(frozen=True)
class Turbine:
    """A turbine's hourly means of the channels read, on the farm's axis, NaN where unread."""

    name: str
    values: Floats

    def before(self, start: int) -> Turbine:
        """The turbine as a new system from ``start``: every hour before it unknown."""
        values = self.values.copy()
        values[:start] = np.nan
        return Turbine(self.name, values)


def loaded(directory: Path, turbines: Sequence[str]) -> list[Turbine]:
    """The turbines' hourly means of the window's channels, on one axis to the end read.

    Raises:
        ValueError: If a turbine lacks a channel of the window or holds an hour past the years
            read.
    """
    end = hour_of(READ_UNTIL)
    out = []
    for name in turbines:
        with np.load(directory / f"hourly-{name}.npz") as arrays:
            channels = [str(c) for c in arrays["channels"]]
            hours, means = arrays["hours"], arrays["means"].astype(np.float64)
        missing = [c for c in CHANNELS if c not in channels]
        if missing:
            raise ValueError(f"turbine {name} holds no {missing}")
        if hours.max() >= end:
            raise ValueError(f"turbine {name} holds an hour past {READ_UNTIL}")
        values = np.full((end, len(CHANNELS)), np.nan)
        values[hours] = means[:, [channels.index(c) for c in CHANNELS]]
        out.append(Turbine(name, values))
    return out
