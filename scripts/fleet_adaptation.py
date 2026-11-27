"""What the published forecaster is fine-tuned on when a turbine or a farm is read as new.

A fit is trained on series in the form the forecaster answers the cells in: per turbine, the
seventeen temperatures as the target and the six operating conditions as covariates known in the
forecast hour, as ``fleet_chronos2_answers.nbm_input`` hands them; per mission of the satellite
telemetry, its channels in groups of at most seventeen as the target alone. Every series ends
before one cut, the hour the fit's cells start after. Two kinds of fit read a system as new:

- a turbine of a known farm: the farm's other turbines, the turbines of a fold left out, so
  that each fit answers the cells of the turbines it never read; in one arm another farm's
  turbines beside them;
- a farm: the other farm's turbines, and in one arm the missions beside them.

A cell is answered only where the hour's six conditions all read, and a farm's early years may
lack one: a fit may read every span, or only those cut from the hours whose conditions all read
(``complete``).

A cell is answered from at most the thirty days since its turbine's start, so a history is cut
into spans of thirty days and the hour after, each a series: a training draw then reads the
hours from a span's start, as a cell reads those from its turbine's start, and the forecaster's
scaling sees contexts of the lengths it answers from. The forecaster draws a series uniformly at
each step, so the missions' share of the draws is their share of the series; their groups of at
most seventeen channels keep it near their share of the loss too. Each fit's record names what
it read, and ``refused`` checks a fit against the cells it is asked to answer. Only numpy and
the standard library are imported, so the forecaster's own environment builds the same series
the tests check.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.fleet_hours import (
    CHANNELS,
    CONDITIONS,
    HELD_UNREAD,
    HOURS_PER_DAY,
    READ_UNTIL,
    TARGETS,
    Turbine,
    hour_of,
    parsed_axis,
)

Floats = NDArray[np.float64]
Key = tuple[int, str, int]
# A series of the forecaster's training inputs: the target, and covariates by name.
Series = dict[str, Any]
# The most hours a cell's context holds since its start: the last verdict day runs to day 30.
SPAN = 30 * HOURS_PER_DAY
# A span reading fewer target hours than one cell's 47 and its hour teaches next to nothing.
LEAST_READ = 48


def folds(names: Sequence[str], count: int) -> list[tuple[str, ...]]:
    """The names, sorted, dealt in turn into ``count`` folds.

    Raises:
        ValueError: If there are fewer names than folds, or no fold.
    """
    if count < 1 or count > len(names):
        raise ValueError(f"{len(names)} turbines cannot be dealt into {count} folds")
    ordered = sorted(names)
    return [tuple(ordered[n::count]) for n in range(count)]


def scope_of(
    farm: str,
    names: Sequence[str],
    *,
    fold: int | None = None,
    of: int | None = None,
    other: str | None = None,
    beside: str | None = None,
    missions: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The farm's turbines a fit answers, and those it must not read, as ``<farm>/<name>``.

    A fit reads the farm without fold ``fold`` of ``of``, with the farm ``beside`` as well when
    named, and answers the fold; or it reads the farm ``other`` alone, with the missions when
    asked, and answers every turbine of the farm.

    Raises:
        ValueError: If neither a fold nor another farm is named, or both; if a farm is read
            beside no fold; if the missions are read beside a fold; or if the farm itself is
            named as the other one.
    """
    if farm in (other, beside):
        raise ValueError(f"{farm} is the farm answered, not another one")
    if other:
        if fold is not None or of is not None or beside:
            raise ValueError("a fit reads either a fold's farm or another farm alone, not both")
        return tuple(names), ()
    if missions:
        raise ValueError("missions are read beside another farm alone")
    if fold is None or of is None:
        raise ValueError("name a fold of the farm, or another farm")
    dealt = folds(names, of)
    if not 0 <= fold < of:
        raise ValueError(f"the farm is dealt into folds 0 to {of - 1}, not {fold}")
    return dealt[fold], tuple(f"{farm}/{name}" for name in dealt[fold])


def spans(read: NDArray[np.bool_]) -> list[slice]:
    """The spans of ``SPAN`` hours and the hour after, from the first hour ``read`` on.

    A span is kept where it reads at least ``LEAST_READ`` hours.
    """
    hours = np.flatnonzero(read)
    if not len(hours):
        return []
    out = []
    for first in range(int(hours[0]), len(read), SPAN + 1):
        kept = slice(first, min(first + SPAN + 1, len(read)))
        if read[kept].sum() >= LEAST_READ:
            out.append(kept)
    return out


def turbine_series(turbine: Turbine, until: int, *, complete: bool = False) -> list[Series]:
    """A turbine's spans before ``until`` in the normal-behaviour form.

    The conditions are marked known in the forecast hour; in training the forecaster takes
    their value there from the series itself. With ``complete``, the spans are cut from the
    hours whose conditions all read: from the first of them on, and kept by their count.
    """
    values = turbine.values[:until, : len(CHANNELS)]
    read = np.isfinite(values[:, : len(TARGETS)]).any(axis=1)
    if complete:
        read &= np.isfinite(values[:, list(CONDITIONS)]).all(axis=1)
    return [
        {
            "target": values[span, : len(TARGETS)].T.copy(),
            "past_covariates": {f"s{i}": values[span, i].copy() for i in CONDITIONS},
            "future_covariates": {f"s{i}": None for i in CONDITIONS},
        }
        for span in spans(read)
    ]


def mission_series(values: Floats) -> list[Series]:
    """A mission's spans, its channels in turn in groups of at most seventeen.

    ``values`` is ``[hours, channels]``; each group's span is a target with no covariate.
    """
    out = []
    for first in range(0, values.shape[1], len(TARGETS)):
        group = values[:, first : first + len(TARGETS)]
        out += [{"target": group[span].T.copy()} for span in spans(np.isfinite(group).any(axis=1))]
    return out


@dataclass(frozen=True)
class FitRecord:
    """What one fit read, written beside it, and the cells it may answer.

    Attributes:
        name: The fit's directory name.
        arm: The answer's name, before the rung's suffix.
        farm: The farm whose cells it answers.
        answers: The turbines whose cells it answers; none of them read.
        turbines: The turbines it read, as ``<farm>/<name>``.
        missions: The missions it read beside them.
        share: The missions' share of the series.
        turbine_series: The turbines' series.
        mission_series: The missions' series, drawn by ``seed``.
        until: The first day it read nothing of.
        model: The forecaster adapted.
        mode: ``lora``, low-rank adapters, or ``full``, every weight.
        rank: The low-rank adapters' rank, or 0 under ``full``.
        steps: Optimiser steps.
        learning_rate: The peak learning rate.
        schedule: ``linear`` to zero, or ``constant``.
        context: The most hours of context a training draw reads.
        batch: Rows per step, each target and covariate of a series a row: a turbine's span is
            23, so 128 draws about six spans.
        save_every: The steps between adapters kept on the way, or 0.
        seed: Seeds the draws and the adapters' start.
        complete: Whether the turbines' spans are cut from the hours whose conditions all read.
        compiled: Whether the model was compiled for the card; the same model, run faster.
        loader_workers: Processes drawing the batches beside the trainer's, 0 for none; with any,
            each draws from its own seeded stream, so the draws differ from those of none.
    """

    name: str
    arm: str
    farm: str
    answers: tuple[str, ...]
    turbines: tuple[str, ...]
    missions: tuple[str, ...]
    share: float
    turbine_series: int
    mission_series: int
    until: str
    model: str
    mode: str
    rank: int
    steps: int
    learning_rate: float
    schedule: str
    context: int
    batch: int
    save_every: int
    seed: int
    complete: bool = False
    compiled: bool = False
    loader_workers: int = 0

    def write(self, directory: Path) -> None:
        (directory / "fit.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def read(cls, directory: Path) -> FitRecord:
        data = json.loads((directory / "fit.json").read_text())
        for field in ("answers", "turbines", "missions"):
            data[field] = tuple(data[field])
        return cls(**data)


@dataclass(frozen=True)
class TrainingSet:
    """The series a fit reads, the turbines among them as ``<farm>/<name>``, and the missions'."""

    series: list[Series]
    turbines: tuple[str, ...]
    turbine_series: int
    mission_series: int


def training_series(
    farms: Mapping[str, Sequence[Turbine]],
    *,
    left_out: Collection[str],
    until: date,
    missions: Mapping[str, Floats] | None = None,
    share: float = 0.0,
    seed: int = 1,
    complete: bool = False,
) -> TrainingSet:
    """Every series a fit reads.

    ``farms`` holds each farm's turbines by the farm's name; ``left_out`` names the turbines
    that must not be read, as ``<farm>/<name>``. The missions' series, drawn by ``seed``, are
    ``share`` of all. With ``complete``, see ``turbine_series``.

    Raises:
        ValueError: If the cut lies past the years read, a turbine left out is not loaded, a
            turbine held unread is handed, the share is outside (0, 1) with missions, or the
            missions hold too few series for it.
    """
    if until > READ_UNTIL:
        raise ValueError(f"a fit may read nothing from {READ_UNTIL}, not until {until}")
    loaded = {f"{farm}/{t.name}" for farm, turbines in farms.items() for t in turbines}
    absent = sorted(set(left_out) - loaded)
    if absent:
        raise ValueError(f"turbines {absent} are left out but not loaded")
    cut = hour_of(until)
    series: list[Series] = []
    read: list[str] = []
    for farm, turbines in sorted(farms.items()):
        for turbine in turbines:
            if turbine.name in HELD_UNREAD:
                raise ValueError(f"turbine {turbine.name} is held unread")
            if f"{farm}/{turbine.name}" in left_out:
                continue
            made = turbine_series(turbine, cut, complete=complete)
            if made:
                series += made
                read.append(f"{farm}/{turbine.name}")
    own = len(series)
    if not missions:
        return TrainingSet(series, tuple(read), own, 0)
    if not 0 < share < 1:
        raise ValueError(f"the missions cannot take a share of {share}")
    pool = [one for name in sorted(missions) for one in mission_series(missions[name])]
    wanted = round(share * own / (1 - share))
    if wanted > len(pool):
        raise ValueError(f"the missions hold {len(pool)} series, not the {wanted} wanted")
    drawn = np.sort(np.random.default_rng(seed).choice(len(pool), wanted, replace=False))
    return TrainingSet(series + [pool[n] for n in drawn], tuple(read), own, wanted)


def refused(record: FitRecord, cells: Iterable[Key]) -> list[str]:
    """Why the fit may not answer the cells: any it read, or any starting before its cut."""
    read = {
        name for farm, _, name in (t.partition("/") for t in record.turbines) if farm == record.farm
    }
    cut = date.fromisoformat(record.until)
    out = []
    for axis in sorted({k[1] for k in cells}):
        turbine, start = parsed_axis(axis)
        if turbine not in record.answers:
            out.append(f"{axis}: {record.name} does not answer turbine {turbine}")
        if turbine in read:
            out.append(f"{axis}: {record.name} read turbine {turbine}")
        if start < cut:
            out.append(f"{axis}: starts before {record.name}'s cut {cut}")
    return out
