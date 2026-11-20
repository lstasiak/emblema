r"""Each satellite channel's mean and count of readings per hour, from a publication of hours.

The raw ESA telemetry is published as windows of one hour every hour, so a window's tokens of a
channel are that channel's readings in the hour, in the publication's normalised units. This
writes, for one mission, every published hour's mean and count of readings per channel: the
series the cold-start forecasts read. Nothing is fitted. A window is the hour it starts in: a
month's windows start at the month's start, and a mission's first month at its first reading's
whole second, seconds into the axis's first hour. An hour with no reading of any channel holds
no window, and is absent; months named in ``--frozen`` are not read.

    uv run --env-file .env.r2 scripts/telemetry_hourly_means.py --manifest KEY CHECKSUM \\
        --mission ESA-Mission2 [--frozen UNITS.txt] --out DIR

Writes ``DIR/hourly-<mission>.npz`` (``hours``, the hour since the missions' origin each row
starts in; ``channels``, the names; ``means`` per hour and channel, NaN where it read nothing;
``counts``) and ``DIR/hourly-<mission>.json``: the manifest, the months left out and the
furthest into its hour a window starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the package is installed, and the store is reached through it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
# The furthest into its hour a window may start, in hours: a minute.
LATEST_START = 1.0 / 60.0


@dataclass(frozen=True)
class Hour:
    """One published window: the hour it starts at and its tokens' channels and values."""

    start: float
    channel_ids: Sequence[int]
    values: Sequence[float]


def hourly_means(
    hours: Iterable[Hour], columns: Mapping[int, int]
) -> tuple[Ints, Floats, NDArray[np.int32], float]:
    """Each hour's mean and count per channel, the channels in ``columns``' order.

    ``columns`` maps a channel identifier to its column. A window is the hour it starts in, at
    most a minute into it, as windows of an hour every hour do; the furthest in is returned.

    Raises:
        ValueError: If a token's channel has no column, a window starts more than a minute into
            its hour, or two start in the same hour.
    """
    width = len(columns)
    starts: list[int] = []
    sums: list[Floats] = []
    counts: list[NDArray[np.int32]] = []
    lookup = np.full(max(columns, default=0) + 1, -1, dtype=np.int64)
    for channel_id, column in columns.items():
        lookup[channel_id] = column
    furthest = 0.0
    for hour in hours:
        whole = int(np.floor(hour.start))
        into = hour.start - whole
        if into > LATEST_START:
            raise ValueError(f"a window starts at {hour.start}, more than a minute into its hour")
        furthest = max(furthest, into)
        ids = np.asarray(hour.channel_ids, dtype=np.int64)
        if ids.size and (ids.max() >= lookup.size or (lookup[ids] < 0).any()):
            raise ValueError(f"the hour at {hour.start} holds a channel with no column")
        at = lookup[ids]
        sums.append(np.bincount(at, weights=np.asarray(hour.values, np.float64), minlength=width))
        counts.append(np.bincount(at, minlength=width).astype(np.int32))
        starts.append(whole)
    order = np.argsort(np.asarray(starts, dtype=np.int64), kind="stable")
    ordered = np.asarray(starts, dtype=np.int64)[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("two windows start at the same hour")
    total = np.asarray(sums, dtype=np.float64).reshape(-1, width)[order]
    count = np.asarray(counts, dtype=np.int32).reshape(-1, width)[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(count > 0, total / count, np.nan)
    return ordered, means, count, furthest


def mission_columns(
    channels: Iterable[tuple[int, str, str]], corpus: str, mission: str
) -> dict[int, int]:
    """The columns of a mission's channels, by identifier, in identifier order.

    ``channels`` are the manifest's (identifier, corpus, name); a mission's channels are named
    ``<mission>/channel_<n>``.
    """
    own = sorted(
        channel_id
        for channel_id, owner, name in channels
        if owner == corpus and name.startswith(f"{mission}/")
    )
    return {channel_id: column for column, channel_id in enumerate(own)}


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the store
    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.shared.kernel.artifacts import ArtifactRef
    from emblema.shared.kernel.checksums import Checksum

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", nargs=2, required=True, metavar=("KEY", "CHECKSUM"))
    parser.add_argument("--mission", required=True)
    parser.add_argument("--frozen", type=Path, help="units never read, one per line")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    settings = Settings()
    blocks = PublishedCorpusBlocks(configured_store(settings), settings.require_worker().workspace)
    manifest = blocks.manifest_of(
        ArtifactRef(arguments.manifest[0], Checksum.parse(arguments.manifest[1]))
    )
    block = blocks.block_of(manifest)
    frozen = set(arguments.frozen.read_text().split()) if arguments.frozen is not None else set()
    read = [
        index
        for index, unit in enumerate(manifest.units)
        if unit.startswith(f"{arguments.mission}/") and unit not in frozen
    ]
    if not read:
        raise SystemExit(f"the publication holds no month of {arguments.mission} to read")
    columns = mission_columns(
        ((c.channel_id, c.corpus, c.channel) for c in manifest.channels),
        manifest.corpus,
        arguments.mission,
    )
    _, starts, _ = block.places_of(read)
    windows = block.of_units(read)
    hours, means, counts, furthest = hourly_means(
        (
            Hour(float(start), window.channel_ids, window.values)
            for start, window in zip(starts.tolist(), windows, strict=True)
        ),
        columns,
    )
    names = {c.channel_id: c.channel for c in manifest.channels}
    arguments.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out / f"hourly-{arguments.mission}.npz",
        hours=hours,
        channels=np.asarray([names[i] for i in columns]),
        means=means.astype(np.float32),
        counts=counts,
    )
    (arguments.out / f"hourly-{arguments.mission}.json").write_text(
        json.dumps(
            {
                "manifest": list(arguments.manifest),
                "corpus": manifest.corpus,
                "mission": arguments.mission,
                "months_read": len(read),
                "months_frozen": sorted(
                    unit
                    for unit in frozen & set(manifest.units)
                    if unit.startswith(f"{arguments.mission}/")
                ),
                "hours": int(hours.size),
                "latest_start_into_hour": furthest,
                "channels": len(columns),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{arguments.mission}: {hours.size} hours of {len(columns)} channels")


if __name__ == "__main__":  # pragma: no cover
    main()
