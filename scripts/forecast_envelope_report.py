"""Read the first mission through a published forecaster's envelope for each channel's hour.

The departures with no model read each channel's hour against the median of its own previous
hours. Here the hour is read against a published forecaster's answer instead: Chronos-2, given
the 47 hourly means of a window's channels before the hour, answers the hour's mean with its
median and its 10th and 90th percentiles, and a channel departs by the distance of the hour's
mean from that median over the spread between the outer two, scaled to a standard deviation and
floored as the departures are. A window's score is the largest departure over its channels, or
their mean. Beside it, fused: each channel's larger departure, from the envelope or from its own
recent hours over 24 hours of bins, the arm whose reading set the verdict's bar on the
validation months. Both read the hour's mean alone, where the departures read five summaries.

Two commands around the forecaster, which runs in its own environment
(``chronos2_envelope.py``):

- ``contexts``: every loaded window's 47 hourly means per channel before it, and how many of
  those hours hold a reading, beside the context report's run.
- ``measure``: the envelope's and the fused departures of a section's windows, and every cell's
  answers as the departure arms write theirs, each alarm set on the section's fitting side.

The verdict is read by ``telemetry_verdict_measure_report.py`` with these answers as candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.telemetry_channel_local_report import (  # noqa: E402
    BINS,
    HOUR,
    MAX,
    MEAN,
    SECTIONS,
    Arm,
    departures_from,
    window_scores,
)

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

# The hour read and the 47 before it, as the forecast was read on 2026-11-15.
HOURS = 48
# The 90th less the 10th percentile of a normal distribution, in standard deviations, so the
# envelope's spread is on the scale the departures' scaled median absolute deviation is.
SPREAD_OF_DECILES = 2.5631
# A channel is read only when half the hours before it hold a reading, as the departures ask.
LEAST_CONTEXT = Arm(HOUR, HOURS).least_context
ENVELOPE, FUSED = "envelope", "fused"
FAMILIES = (ENVELOPE, FUSED)
DETECTORS = (MAX, MEAN)
# The departure arm fused with the envelope: its mean over channels read the bar's value.
PARTNER = Arm(BINS, 24)
# The hour's mean, the first of the five summaries the departures keep per channel.
MEAN_SUMMARY = 0


def hour_contexts(means: Floats, before: Ints) -> NDArray[np.float32]:
    """``[len(before), rows, hours]``: the channels' hourly means before each window.

    ``means`` is ``[windows, rows]``, ``before`` the windows of the hours before each, oldest
    first, -1 where none was loaded; such an hour is missing in every channel.
    """
    gathered = np.where((before >= 0)[:, :, None], means[np.maximum(before, 0)], np.nan)
    return np.ascontiguousarray(gathered.transpose(0, 2, 1), dtype=np.float32)


def envelope_departures(
    now: Floats, envelope: Floats, loaded: Ints, floor: Floats, *, least_context: int
) -> Floats:
    """``[windows, rows]``: each channel's departure of the hour's mean from the envelope.

    ``envelope`` is ``[windows, rows, 3]``, the 10th, 50th and 90th percentiles; ``floor`` the
    least spread per channel. A channel absent from the hour, without an envelope, or with fewer
    than ``least_context`` hours read before it, departs nought.
    """
    spread = (envelope[..., 2] - envelope[..., 0]) / SPREAD_OF_DECILES
    return departures_from(
        now[..., None, None],
        envelope[..., 1:2],
        spread[..., None],
        loaded,
        floor[:, None],
        least_context=least_context,
    )


def fused_departures(envelope: Floats, history: Floats) -> Floats:
    """Each channel's larger departure, from the envelope or from its own recent hours.

    Raises:
        ValueError: If the two do not read the same windows and channels.
    """
    if envelope.shape != history.shape:
        raise ValueError(f"the envelope reads {envelope.shape}, the history {history.shape}")
    fused: Floats = np.maximum(envelope, history)
    return fused


def cell_name(section: str, family: str, detector: str) -> str:
    return f"{section}-{family}-{detector}"


def contexts(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every loaded window's hourly means before it, for the forecaster; written once."""
    from scripts.taught_fine_tuning_pilot import bench_of
    from scripts.telemetry_channel_local_report import Departures
    from scripts.telemetry_context_report import Reader

    out: Path = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    reader = Reader(bench_of(arguments), arguments.context)
    means = Departures(reader.cells, arguments.departures).summaries()[..., MEAN_SUMMARY]
    held = np.lib.format.open_memmap(
        out / "contexts.npy", mode="w+", dtype=np.float32, shape=(*means.shape, HOURS - 1)
    )
    loaded = np.zeros(means.shape, dtype=np.int64)
    for start in range(0, len(means), 4096):
        indices = list(range(start, min(start + 4096, len(means))))
        part = hour_contexts(means, reader.cells.previous(indices, HOURS))
        held[start : start + len(indices)] = part
        loaded[start : start + len(indices)] = np.isfinite(part).sum(axis=2)
    held.flush()
    np.save(out / "loaded.npy", loaded)
    print(f"{len(means)} windows, {int((loaded > 0).sum())} channel hours with a context")


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - environment
    """Every cell of a section: the envelope's and the fused departures, each under its alarm."""
    from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
    from emblema.evaluation.contracts.identifiers import CandidateRef
    from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
    from scripts.taught_fine_tuning_pilot import (
        bench_of,
        committed_tree,
        refuse_another_commit,
        write_answers,
    )
    from scripts.telemetry_channel_local_report import Departures
    from scripts.telemetry_context_report import Reader, refuse_other_conditions

    out: Path = arguments.out
    (out / "answers").mkdir(parents=True, exist_ok=True)
    commit = "trial" if arguments.trial else committed_tree()
    refuse_another_commit(out / "conditions.json", commit)
    departed = json.loads((arguments.departures.parent / "conditions.json").read_text())
    if (departed["select_from"], departed["context"]) != (
        arguments.select_from,
        str(arguments.context),
    ):
        raise SystemExit("the departures were measured on another context or selection")
    if not (arguments.departures / "summaries.npy").exists():
        raise SystemExit("the departures' summaries are missing; the departures run keeps them")
    conditions = {
        "commit": commit,
        "section": arguments.section,
        "select_from": arguments.select_from,
        "context": str(arguments.context),
        "departures": str(arguments.departures),
        "contexts": str(arguments.contexts),
        "envelope": str(arguments.envelope),
        "partner": PARTNER.name,
        "least_context": LEAST_CONTEXT,
    }
    refuse_other_conditions(out / "conditions.json", conditions)
    reader = Reader(bench_of(arguments), arguments.context)
    fit, scored, _ = reader.sides(arguments.section, arguments.select_from)
    departures_run = Departures(reader.cells, arguments.departures)
    summaries = departures_run.summaries()
    envelope = np.load(arguments.envelope)
    loaded = np.load(arguments.contexts / "loaded.npy")
    if envelope.shape[:2] != summaries.shape[:2] or loaded.shape != summaries.shape[:2]:
        raise SystemExit("the envelope or the contexts were laid over other windows")
    floor = departures_run.floor(fit)[:, MEAN_SUMMARY]
    now = summaries[..., MEAN_SUMMARY]
    departures = {
        ENVELOPE: envelope_departures(now, envelope, loaded, floor, least_context=LEAST_CONTEXT)
    }
    history = np.load(arguments.departures / f"{arguments.section}-{PARTNER.name}.npy")
    departures[FUSED] = fused_departures(departures[ENVELOPE], history)
    present = departures_run.present()
    contamination = (
        reader.catalogue.arm_of(CandidateRef("untrained_knn@distance=whitened,reference=unit"))
        .recipe(seed=1)
        .contamination
    )
    units_fit = [str(reader.windows[i].window.unit) for i in fit]
    units_scored = [str(reader.windows[i].window.unit) for i in scored]
    fit_array = np.asarray(fit, dtype=np.int64)
    scored_array = np.asarray(scored, dtype=np.int64)
    for family in FAMILIES:
        for detector in DETECTORS:
            scores = window_scores(departures[family], present, detector)
            alarm = ScoreAlarm.set_over(
                scores[fit_array], contamination=contamination, units=units_fit
            )
            above = alarm.above(scores[scored_array], units=units_scored)
            write_answers(
                out / "answers" / f"{cell_name(arguments.section, family, detector)}.csv",
                (
                    WindowPrediction(
                        window=reader.windows[i].window,
                        target=reader.windows[i].target,
                        predicted=float(s),
                    )
                    for i, s in zip(scored, above.tolist(), strict=True)
                ),
            )
    counted = {"fitting_windows": len(fit), "scored_windows": len(scored)}
    (out / "conditions.json").write_text(json.dumps({**conditions, **counted}, indent=2) + "\n")
    print(f"{arguments.section}: {len(FAMILIES) * len(DETECTORS)} cells of {len(scored)} windows")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--task", required=True)
    shared.add_argument("--task-id", required=True)
    shared.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    shared.add_argument(
        "--weights", nargs=3, action="append", default=[], metavar=("NAME", "KEY", "CHECKSUM")
    )
    shared.add_argument("--context", type=Path, required=True, help="the context report's run")
    shared.add_argument("--departures", type=Path, required=True, help="the departures' arrays")
    shared.add_argument("--device", default="cpu")
    shared.add_argument("--out", type=Path, required=True)
    commands.add_parser("contexts", parents=[shared], help="the forecaster's contexts")
    measuring = commands.add_parser("measure", parents=[shared], help="a section's cells")
    measuring.add_argument("--section", choices=SECTIONS, required=True)
    measuring.add_argument("--select-from", required=True, help="the first month that chooses")
    measuring.add_argument("--contexts", type=Path, required=True, help="what contexts wrote")
    measuring.add_argument("--envelope", type=Path, required=True, help="the forecaster's npy")
    measuring.add_argument("--trial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
    arguments = parse_arguments(argv)
    {"contexts": contexts, "measure": measure}[arguments.command](arguments)


if __name__ == "__main__":  # pragma: no cover
    main()
