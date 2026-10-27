"""The ZTF table matches a quadrant's curves across bands and writes the drawn sources' readings.

Held to a quadrant written in the release's layout, one Parquet file per band with the epochs as
list columns, whose sources, matches and good epochs are known.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.ztf_light_curve_table import (
    BANDS,
    Curves,
    drawn,
    good_epochs,
    quadrants,
    sources,
    table,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ARCSECOND = 1.0 / 3600.0
QUADRANT = "000100_c01_q1"


def curves(*positions: tuple[float, float]) -> Curves:
    ra = np.asarray([p[0] for p in positions])
    dec = np.asarray([p[1] for p in positions])
    count = len(positions)
    return Curves(np.arange(count, dtype=np.int64), ra, dec, np.full(count, 60, dtype=np.int64))


def test_curves_each_others_nearest_within_the_radius_are_one_source() -> None:
    found, ambiguous = sources(
        {
            "g": curves((10.0, 20.0), (10.1, 20.0)),
            "r": curves((10.0 + 0.5 * ARCSECOND, 20.0)),
            "i": curves((10.0, 20.0 + 0.8 * ARCSECOND)),
        },
        radius=1.5,
    )

    assert sorted(found, key=len) == [{"g": 1}, {"g": 0, "r": 0, "i": 0}]
    assert ambiguous == 0


def test_a_curve_beyond_the_radius_stays_a_source_of_its_own_and_a_band_may_hold_none() -> None:
    found, _ = sources(
        {"g": curves((10.0, 20.0)), "r": curves((10.0, 20.0 + 2 * ARCSECOND)), "i": curves()},
        1.5,
    )

    assert sorted(found, key=lambda s: list(s)) == [{"g": 0}, {"r": 0}]


def test_a_source_holding_two_curves_of_one_band_is_ambiguous_and_dropped() -> None:
    found, ambiguous = sources(
        {
            "g": curves((10.0, 20.0), (10.0, 20.0 + 1.2 * ARCSECOND)),
            "r": curves((10.0, 20.0 + 0.2 * ARCSECOND)),
            "i": curves((10.0, 20.0 + 1.0 * ARCSECOND)),
        },
        radius=1.5,
    )

    assert found == []
    assert ambiguous == 1


def test_a_good_epoch_is_unflagged_with_a_positive_error_and_a_magnitude() -> None:
    good = good_epochs(
        np.asarray([18.0, 18.0, 18.0, np.nan]),
        np.asarray([0.1, -0.1, 0.1, 0.1]),
        np.asarray([0, 0, 32768, 0], dtype=np.int64),
    )

    assert good.tolist() == [True, False, False, False]


def test_only_quadrants_holding_every_band_are_drawn(tmp_path: Path) -> None:
    lines = [
        f"{'0' * 32}  ./0/field000100/ztf_000100_z{band}_c01_q{q}_dr24.parquet"
        for q, held in ((1, "gri"), (2, "gr"), (3, "gri"))
        for band in held
    ]
    (tmp_path / "checksum.md5").write_text("\n".join([*lines, "junk"]) + "\n")

    held = quadrants(tmp_path / "checksum.md5")

    assert set(held) == {"000100_c01_q1", "000100_c01_q2", "000100_c01_q3"}
    assert held["000100_c01_q1"]["i"] == "0/field000100/ztf_000100_zi_c01_q1_dr24.parquet"
    assert set(drawn(held, 5, seed=1)) == {"000100_c01_q1", "000100_c01_q3"}
    assert drawn(held, 1, seed=1) == drawn(held, 1, seed=1)


def write_band(path: Path, rows: list[tuple[int, float, float, list[float], list[int]]]) -> None:
    """Rows of objectid, position, epochs and their flags; every magnitude 18, every error 0.1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "objectid": [r[0] for r in rows],
                "objra": [r[1] for r in rows],
                "objdec": [r[2] for r in rows],
                "hmjd": [r[3] for r in rows],
                "mag": [[18.0] * len(r[3]) for r in rows],
                "magerr": [[0.1] * len(r[3]) for r in rows],
                "catflags": [r[4] for r in rows],
            }
        ),
        path,
    )


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    near = 0.3 * ARCSECOND
    epochs = [float(t) for t in range(60)]
    files = {band: f"0/field000100/ztf_000100_z{band}_c01_q1_dr24.parquet" for band in BANDS}
    write_band(
        tmp_path / files["g"],
        [(11, 10.0, 20.0, epochs[:30], [0] * 30), (12, 10.1, 20.0, epochs, [0] * 59 + [32768])],
    )
    write_band(tmp_path / files["r"], [(21, 10.0 + near, 20.0, epochs[:30], [0] * 30)])
    write_band(tmp_path / files["i"], [(31, 10.2, 20.0, epochs[:10], [0] * 10)])
    return tmp_path, {QUADRANT: files}


def test_the_table_writes_the_good_epochs_of_the_sources_in_the_frame(
    written: tuple[Path, dict[str, dict[str, str]]], tmp_path: Path
) -> None:
    root, held = written
    out = tmp_path / "out"

    [row] = table(root, held, [QUADRANT], out, radius=1.5, least=50, units=10, seed=1)

    assert (row.curves_g, row.curves_r, row.curves_i) == (2, 1, 1)
    assert (row.sources, row.ambiguous, row.in_frame, row.frame_epochs) == (3, 0, 2, 119)
    with (out / "readings.csv").open() as file:
        readings = list(csv.DictReader(file))
    by_unit: dict[tuple[str, str], int] = {}
    for reading in readings:
        key = (reading["unit"], reading["channel"])
        by_unit[key] = by_unit.get(key, 0) + 1
    assert by_unit == {
        (f"{QUADRANT}/11", "g"): 30,
        (f"{QUADRANT}/11", "r"): 30,
        (f"{QUADRANT}/12", "g"): 59,
    }
    assert (out / "frame.csv").exists()


def test_the_units_written_are_the_raw_audits_draw_from_the_frame(
    written: tuple[Path, dict[str, dict[str, str]]], tmp_path: Path
) -> None:
    root, held = written
    out = tmp_path / "out"

    table(root, held, [QUADRANT], out, radius=1.5, least=50, units=1, seed=1)

    with (out / "readings.csv").open() as file:
        units = {reading["unit"] for reading in csv.DictReader(file)}
    assert len(units) == 1
