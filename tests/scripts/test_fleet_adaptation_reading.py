"""The fine-tuned forecaster's pilot read: its pairs, its bars and the files it joins."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.fleet_adaptation_reading import WITHOUT_THE_FARM, compared, month
from scripts.fleet_cold_start_cells import VERDICT_DAYS
from scripts.fleet_hours import axis_of

STARTS = (date(2020, 3, 1), date(2021, 1, 1), date(2022, 5, 1))


def written(directory: Path, kind: str, number: int, answers: dict[str, float]) -> None:
    with (directory / f"{kind}-farm-N{number}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        extra = ("target_mean",) if kind == "answers" else ()
        writer.writerow(("window", "axis", "channel", *extra, *(f"mean_{a}" for a in answers)))
        rng = np.random.default_rng(number)
        for start in STARTS:
            for turbine in ("1", "2"):
                for channel in (0, 3):
                    noise = rng.normal() * 0.01
                    shown = (f"{v + noise:.6g}" for v in answers.values())
                    target = ("0",) if kind == "answers" else ()
                    writer.writerow(
                        (100 + number, axis_of(turbine, start), channel, *target, *shown)
                    )


def test_the_month_joins_every_kind_of_each_verdict_day_and_reads_the_pairs(tmp_path: Path) -> None:
    for number in VERDICT_DAYS:
        written(tmp_path, "answers", number, {"last_hour": 1.0, "nbm_ridge": 0.8})
        written(tmp_path, "trees-pilot", number, {"other_farm_nbm_trees_until2020": 0.6})
        written(tmp_path, "adapted_other", number, {"adapted_other_nbm": 0.3})

    day = month(tmp_path, "farm", ["answers", "trees-pilot", "adapted_other"])
    rows = compared(
        day,
        [
            ("adapted_other_nbm", f"{WITHOUT_THE_FARM}, other farm's trees"),
            ("adapted_other_nbm", "last_hour"),
        ],
        resamples=200,
        seed=1,
    )

    assert len(day.keys) == len(VERDICT_DAYS) * len(STARTS) * 4
    assert [r["against"] for r in rows] == [
        f"{WITHOUT_THE_FARM}, other farm's trees, best of 3: other_farm_nbm_trees_until2020",
        "last_hour",
    ]
    assert rows[0]["ratio"] == pytest.approx(0.25, rel=0.15)
    assert rows[1]["ratio"] == pytest.approx(0.09, rel=0.2)
    assert {r["level"] for r in rows} == {"descriptive"}


def test_a_missing_file_or_an_answer_no_file_gives_is_refused(tmp_path: Path) -> None:
    for number in VERDICT_DAYS:
        written(tmp_path, "answers", number, {"last_hour": 1.0})
    day = month(tmp_path, "farm", ["answers"])

    with pytest.raises(ValueError, match="not written"):
        month(tmp_path, "farm", ["answers", "adapted_fleet"])
    with pytest.raises(ValueError, match="no file answers"):
        compared(day, [("adapted_fleet_nbm", "last_hour")], resamples=10, seed=1)
    with pytest.raises(ValueError, match=r"no file answers \['other_farm_nbm_trees_until2020'\]"):
        compared(
            day, [("last_hour", f"{WITHOUT_THE_FARM}, other farm's trees")], resamples=10, seed=1
        )
