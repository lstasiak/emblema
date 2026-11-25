"""The fine-tuned forecaster's pilot read: its pairs, its bars and the files it joins."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.cold_start_reading import Day, pooled
from scripts.fleet_adaptation_reading import (
    WITHOUT_THE_FARM,
    compared,
    verdict_days,
    widened,
    with_means,
)
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

    day = pooled(verdict_days(tmp_path, "farm", ["answers", "trees-pilot", "adapted_other"]))
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
    day = pooled(verdict_days(tmp_path, "farm", ["answers"]))

    with pytest.raises(ValueError, match="not written"):
        verdict_days(tmp_path, "farm", ["answers", "adapted_fleet"])
    with pytest.raises(ValueError, match="no file answers"):
        compared(day, [("adapted_fleet_nbm", "last_hour")], resamples=10, seed=1)
    with pytest.raises(ValueError, match=r"no file answers \['other_farm_nbm_trees_until2020'\]"):
        compared(
            day, [("last_hour", f"{WITHOUT_THE_FARM}, other farm's trees")], resamples=10, seed=1
        )


def test_a_reading_names_bars_and_means_of_its_own_and_reads_a_day_alone(tmp_path: Path) -> None:
    for number in VERDICT_DAYS:
        written(tmp_path, "answers", number, {"last_hour": 1.0})
        written(tmp_path, "trees-tuned", number, {"trees": 0.6, "trees_ridge": 0.4})
        written(tmp_path, "adapted", number, {"seed1_nbm": 0.2, "seed2_nbm": 0.4})
    kinds = ["answers", "trees-tuned", "adapted"]
    days = [
        with_means(d, {"both_nbm": ("seed1_nbm", "seed2_nbm")})
        for d in verdict_days(tmp_path, "farm", kinds)
    ]
    bars = {"with the history": ("trees", "trees_ridge")}

    rows = compared(
        days[1],
        [("both_nbm", "with the history")],
        resamples=200,
        seed=1,
        bars=bars,
        reading="day 7",
    )

    # The mean of 0.2 and 0.4 against the bar's best member, 0.4: (0.3 / 0.4) squared.
    assert [(r["set"], r["days"], r["against"]) for r in rows] == [
        ("day 7", "7", "with the history, best of 2: trees_ridge")
    ]
    assert rows[0]["ratio"] == pytest.approx(0.5625, rel=0.15)
    assert rows[0]["windows"] == len(STARTS) * 2
    with pytest.raises(ValueError, match=r"no file answers \['trees_boosted'\]"):
        compared(
            days[1],
            [("both_nbm", "wider")],
            resamples=10,
            seed=1,
            bars={"wider": ("trees", "trees_boosted")},
        )
    assert widened([WITHOUT_THE_FARM, "trees"])[-3:] == ("local_nbm_arx", "nbm_ridge", "trees")
    assert "last_hour" in widened([WITHOUT_THE_FARM])
    with pytest.raises(ValueError, match="a bar needs a member"):
        widened([])
    with pytest.raises(ValueError, match="name bars already"):
        compared(days[1], [], resamples=10, seed=1, bars={WITHOUT_THE_FARM: ("trees",)})
    with pytest.raises(ValueError, match="cannot be taken"):
        with_means(days[0], {"both_nbm": ("seed1_nbm",)})
    with pytest.raises(ValueError, match="cannot be taken"):
        with_means(days[0], {"other": ("seed1_nbm", "seed9_nbm")})


def test_a_mean_gives_no_answer_where_one_of_its_answers_gives_none() -> None:
    keys = ((1, "1@2020-03-01", 0), (2, "1@2020-03-01", 0))
    answers = {"a": np.asarray([1.0, 3.0]), "b": np.asarray([2.0, np.nan])}
    day = Day("farm", "2", keys, np.zeros(2), answers)

    both = with_means(day, {"both": ("a", "b")}).answers["both"]

    assert both[0] == 1.5
    assert np.isnan(both[1])
