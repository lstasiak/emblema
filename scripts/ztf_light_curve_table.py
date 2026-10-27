"""Turn the ZTF light curves of drawn field quadrants into the long table the raw audit reads.

ZTF publishes one light curve per object, field quadrant and band, so a source observed in g, r
and i is three curves with three identifiers. This script draws field quadrants that hold all three
bands, matches each quadrant's curves across bands by position, keeps the sources with enough good
epochs and writes the readings of a seeded draw of them as rows of unit, channel and instant.

    uv run scripts/ztf_light_curve_table.py draw --checksums data/raw/ztf/checksum.md5
    uv run scripts/ztf_light_curve_table.py table --root data/raw/ztf \
        --checksums data/raw/ztf/checksum.md5 --out data/report/raw-regime-audit/ztf
    uv run scripts/raw_regime_audit.py --name ztf --root data/report/raw-regime-audit/ztf \
        --pattern readings.csv --unit unit --channel channel --time time

A curve matches the nearest curve of another band within ``--radius`` arcseconds when each is the
other's nearest; matches join into sources, and a source holding two curves of one band is
ambiguous and dropped. A unit is keyed ``<quadrant>/<smallest objectid>``, so the raw audit counts
it within its quadrant. A good epoch has ``catflags`` 0, a positive ``magerr`` and a finite
magnitude. A source enters the frame with at least ``--least`` good epochs over its bands, and the
units are drawn from the frame with the raw audit's own draw, so the audit then reads them all.
``table`` writes ``readings.csv`` (unit, channel, time in days) and ``frame.csv`` (one row per
quadrant: curves per band, sources, ambiguous sources, sources in the frame, their good epochs).
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.kernel.ordering import seeded_rank
from scripts.regime_coverage_report import chosen_keys

BANDS = ("g", "r", "i")
FILE = re.compile(
    r"^(?P<md5>[0-9a-f]{32})\s+\./(?P<path>\d+/field(?P<field>\d+)/"
    r"ztf_\d+_z(?P<band>[gri])_c(?P<ccd>\d+)_q(?P<quadrant>\d)_dr24\.parquet)$"
)
ARCSECONDS_PER_DEGREE = 3600.0
COLUMNS = ("objectid", "objra", "objdec", "hmjd", "mag", "magerr", "catflags")


def quadrants(checksums: Path) -> dict[str, dict[str, str]]:
    """Each field quadrant of the release, by key, with the file of each band it holds."""
    held: dict[str, dict[str, str]] = defaultdict(dict)
    for line in checksums.read_text().splitlines():
        match = FILE.match(line.strip())
        if match:
            key = f"{match['field']}_c{match['ccd']}_q{match['quadrant']}"
            held[key][match["band"]] = match["path"]
    return dict(held)


def drawn(held: Mapping[str, Mapping[str, str]], count: int, seed: int) -> list[str]:
    """``count`` of the quadrants that hold all three bands, by seed, in the order drawn."""
    complete = [key for key, files in held.items() if set(files) == set(BANDS)]
    return sorted(complete, key=lambda key: seeded_rank(seed, key))[:count]


@dataclass(frozen=True)
class Curves:
    """One band's light curves of a quadrant: identifiers, positions and good epochs.

    Attributes:
        ids: Each curve's ``objectid``.
        ra: Its right ascension, degrees.
        dec: Its declination, degrees.
        good: How many of its epochs are good.
    """

    ids: NDArray[np.int64]
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    good: NDArray[np.int64]


def good_epochs(
    mag: NDArray[np.floating], magerr: NDArray[np.floating], catflags: NDArray[np.integer]
) -> NDArray[np.bool_]:
    good: NDArray[np.bool_] = (catflags == 0) & (magerr > 0) & np.isfinite(mag)
    return good


@dataclass(frozen=True)
class Batch:
    """Some rows of a file: per-curve columns, and the epochs' flat arrays cut by ``offsets``."""

    ids: NDArray[np.int64]
    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    offsets: NDArray[np.int64]
    hmjd: NDArray[np.float64]
    good: NDArray[np.bool_]

    def good_per_curve(self) -> NDArray[np.int64]:
        """How many good epochs each curve holds."""
        totals = np.concatenate([[0], np.cumsum(self.good, dtype=np.int64)])
        counts: NDArray[np.int64] = totals[self.offsets[1:]] - totals[self.offsets[:-1]]
        return counts

    def good_times(self, row: int) -> NDArray[np.float64]:
        start, end = self.offsets[row], self.offsets[row + 1]
        return self.hmjd[start:end][self.good[start:end]]


def batches(path: Path) -> Iterator[Batch]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(columns=list(COLUMNS)):
        flat = {name: batch.column(name).flatten().to_numpy() for name in COLUMNS[3:]}
        yield Batch(
            ids=batch.column("objectid").to_numpy().astype(np.int64),
            ra=batch.column("objra").to_numpy().astype(np.float64),
            dec=batch.column("objdec").to_numpy().astype(np.float64),
            offsets=batch.column("hmjd").offsets.to_numpy().astype(np.int64),
            hmjd=flat["hmjd"].astype(np.float64),
            good=good_epochs(flat["mag"], flat["magerr"], flat["catflags"]),
        )


def read_curves(path: Path) -> Curves:
    """The file's curves, its epochs read a batch at a time and only their good count kept."""
    ids: list[NDArray[np.int64]] = []
    ra: list[NDArray[np.float64]] = []
    dec: list[NDArray[np.float64]] = []
    good: list[NDArray[np.int64]] = []
    for batch in batches(path):
        ids.append(batch.ids)
        ra.append(batch.ra)
        dec.append(batch.dec)
        good.append(batch.good_per_curve())
    return Curves(
        np.concatenate(ids), np.concatenate(ra), np.concatenate(dec), np.concatenate(good)
    )


def plane(
    ra: NDArray[np.float64], dec: NDArray[np.float64], centre: tuple[float, float]
) -> NDArray[np.float64]:
    """Positions on the plane tangent at ``centre``, in arcseconds; a quadrant spans < 1 degree."""
    ra0, dec0 = centre
    east = ((ra - ra0 + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec0))
    return np.column_stack([east, dec - dec0]) * ARCSECONDS_PER_DEGREE


def mutual_pairs(
    first: NDArray[np.float64], second: NDArray[np.float64], radius: float
) -> list[tuple[int, int]]:
    """Index pairs that are each other's nearest within ``radius``."""
    from scipy.spatial import cKDTree

    if not len(first) or not len(second):
        return []
    forward_distance, forward = cKDTree(second).query(first, distance_upper_bound=radius)
    _, backward = cKDTree(first).query(second, distance_upper_bound=radius)
    return [
        (i, int(j))
        for i, (d, j) in enumerate(zip(forward_distance, forward, strict=True))
        if np.isfinite(d) and backward[j] == i
    ]


def sources(curves: Mapping[str, Curves], radius: float) -> tuple[list[dict[str, int]], int]:
    """The quadrant's sources as ``{band: curve index}``, and how many were ambiguous."""
    ra = np.concatenate([c.ra for c in curves.values()])
    dec = np.concatenate([c.dec for c in curves.values()])
    centre = (float(np.median(ra)), float(np.median(dec))) if ra.size else (0.0, 0.0)
    points = {band: plane(c.ra, c.dec, centre) for band, c in curves.items()}
    parent = {(band, i): (band, i) for band, c in curves.items() for i in range(len(c.ids))}

    def root(node: tuple[str, int]) -> tuple[str, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    bands = list(curves)
    for at, a in enumerate(bands):
        for b in bands[at + 1 :]:
            for i, j in mutual_pairs(points[a], points[b], radius):
                parent[root((a, i))] = root((b, j))
    members: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for node in parent:
        members[root(node)].append(node)
    found, ambiguous = [], 0
    for part in members.values():
        held = [band for band, _ in part]
        if len(held) != len(set(held)):
            ambiguous += 1
            continue
        found.append(dict(part))
    return found, ambiguous


@dataclass(frozen=True)
class QuadrantRow:
    """One quadrant's matching and frame; a plain record written to ``frame.csv``."""

    quadrant: str
    curves_g: int
    curves_r: int
    curves_i: int
    sources: int
    ambiguous: int
    in_frame: int
    frame_epochs: int


def times_of(path: Path, wanted: set[int]) -> dict[int, NDArray[np.float64]]:
    """The good epochs' instants of the curves named, read in a second pass over the file."""
    names = np.fromiter(wanted, dtype=np.int64)
    found = {}
    for batch in batches(path):
        for row in np.flatnonzero(np.isin(batch.ids, names)):
            found[int(batch.ids[row])] = batch.good_times(int(row))
    return found


@dataclass(frozen=True)
class Unit:
    """A source in the frame: its quadrant, its curve per band and its good epochs."""

    quadrant: str
    curves: Mapping[str, int]
    epochs: int


def table(
    root: Path,
    held: Mapping[str, Mapping[str, str]],
    chosen: Sequence[str],
    out: Path,
    radius: float,
    least: int,
    units: int,
    seed: int,
) -> list[QuadrantRow]:
    """Match every chosen quadrant, draw the units, and write their readings and the frame."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    frame: dict[str, Unit] = {}
    for quadrant in chosen:
        curves = {band: read_curves(root / held[quadrant][band]) for band in BANDS}
        found, ambiguous = sources(curves, radius)
        kept = []
        for source in found:
            epochs = sum(int(curves[band].good[i]) for band, i in source.items())
            if epochs >= least:
                ids = {band: int(curves[band].ids[i]) for band, i in source.items()}
                kept.append(Unit(quadrant, ids, epochs))
        frame |= {f"{quadrant}/{min(unit.curves.values())}": unit for unit in kept}
        rows.append(
            QuadrantRow(
                quadrant=quadrant,
                curves_g=len(curves["g"].ids),
                curves_r=len(curves["r"].ids),
                curves_i=len(curves["i"].ids),
                sources=len(found) + ambiguous,
                ambiguous=ambiguous,
                in_frame=len(kept),
                frame_epochs=sum(unit.epochs for unit in kept),
            )
        )
        print(f"{quadrant}: {len(found)} sources, {ambiguous} ambiguous, {len(kept)} in the frame")
    draw = chosen_keys(frame, units, seed)
    with (out / "readings.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("unit", "channel", "time"))
        for quadrant in chosen:
            for band in BANDS:
                mine = {
                    key: unit.curves[band]
                    for key, unit in frame.items()
                    if key in draw and unit.quadrant == quadrant and band in unit.curves
                }
                if mine:
                    times = times_of(root / held[quadrant][band], set(mine.values()))
                    for key, identifier in sorted(mine.items()):
                        writer.writerows((key, band, repr(float(t))) for t in times[identifier])
    with (out / "frame.csv").open("w", newline="") as file:
        summary = csv.DictWriter(file, fieldnames=[f.name for f in fields(QuadrantRow)])
        summary.writeheader()
        summary.writerows(asdict(row) for row in rows)
    return rows


def main() -> None:  # pragma: no cover - the command line
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("draw", "table"):
        command = commands.add_parser(name)
        command.add_argument("--checksums", type=Path, required=True)
        command.add_argument("--quadrants", type=int, default=5)
        command.add_argument("--seed", type=int, default=1)
    tabling = commands.choices["table"]
    tabling.add_argument("--root", type=Path, required=True, help="where the release's files lie")
    tabling.add_argument("--out", type=Path, required=True)
    tabling.add_argument("--radius", type=float, default=1.5, help="arcseconds")
    tabling.add_argument("--least", type=int, default=50, help="good epochs a source needs")
    tabling.add_argument("--units", type=int, default=100)
    arguments = parser.parse_args()
    held = quadrants(arguments.checksums)
    chosen = drawn(held, arguments.quadrants, arguments.seed)
    if arguments.command == "draw":
        for quadrant in chosen:
            for band in BANDS:
                print(held[quadrant][band])
        return
    table(
        arguments.root,
        held,
        chosen,
        arguments.out,
        arguments.radius,
        arguments.least,
        arguments.units,
        arguments.seed,
    )


if __name__ == "__main__":
    main()
