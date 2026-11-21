"""The turbines' cold-start reading: windows per turbine-hour, starts resampled whole, the sets."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.cold_start_reading import Day
from scripts.fleet_chronos2_answers import RUNGS
from scripts.fleet_cold_start_reading import (
    BARS,
    GROUPS,
    VERDICTS,
    Comparison,
    Windows,
    comparison_rows,
    read,
)
from scripts.fleet_hours import TARGETS, axis_of
from scripts.fleet_tree_answers import TREES
from scripts.local_forecaster_answers import FLAT, FORECASTERS

EARLY, LATE = date(2020, 3, 1), date(2022, 5, 1)


def day(
    keys: list[tuple[int, str, int]], answers: dict[str, list[float]], target: float = 0.0
) -> Day:
    return Day(
        "farm",
        "7",
        tuple(keys),
        np.full(len(keys), target),
        {name: np.asarray(values, dtype=np.float64) for name, values in answers.items()},
    )


def test_a_window_is_one_turbines_hour_and_its_start_is_resampled_whole() -> None:
    keys = [
        (10, axis_of("1", EARLY), 0),
        (10, axis_of("1", EARLY), 1),
        (10, axis_of("2", EARLY), 0),
        (99, axis_of("1", LATE), 0),
    ]
    one = day(keys, {"a": [1.0, 3.0, 5.0, 7.0]})

    windows = Windows.of(one, np.ones(4, dtype=bool))
    weights = windows.weights(500, seed=0)

    assert windows.losses(np.array([1.0, 3.0, 5.0, 7.0])).tolist() == [2.0, 7.0, 5.0]
    assert windows.starts.tolist() == [0, 1, 0]
    assert (weights[:, 0] == weights[:, 2]).all()
    assert (weights[:, 0] + weights[:, 1] == 2).all()


def cells_for(
    starts: list[date], turbines: tuple[str, ...] = ("1", "2"), hour: int = 100
) -> list[tuple[int, str, int]]:
    return [
        (h, axis_of(t, s), c)
        for s in starts
        for t in turbines
        for h in (hour, hour + 1)
        for c in (0, 10)
    ]


def test_the_bar_leaves_out_an_answer_of_few_cells_and_only_verdicts_read_a_level() -> None:
    keys = cells_for([EARLY, LATE, date(2021, 1, 1)])
    rng = np.random.default_rng(3)
    noise = rng.normal(size=len(keys)) * 0.01
    answers = {
        "chronos2_47": list(0.5 + noise),
        "last_hour": list(1.0 + noise),
        "fleet_ridge": list(0.9 + noise),
        "ridge": [np.nan] * (len(keys) // 3) + list(0.1 + noise[len(keys) // 3 :]),
        "oracle_power_ridge": list(0.05 + noise),
        "chronos2_neighbours": list(0.5 + noise),
    }

    rows = comparison_rows(
        day(keys, answers),
        [*VERDICTS[:2], Comparison("oracle_power_ridge", "with the farm")],
        reading="first month",
        levels=True,
        least_gain=0.8,
        resamples=200,
        seed=1,
    )
    by_of = {row["of"]: row for row in rows}

    assert by_of["chronos2_47"]["against"] == "with the farm, best of 2: fleet_ridge"
    assert by_of["chronos2_47"]["ratio"] == pytest.approx(0.25 / 0.81, rel=0.05)
    assert by_of["chronos2_47"]["level"] == "at least the least gain"
    assert by_of["chronos2_47"]["starts"] == 3
    assert str(by_of["chronos2_neighbours"]["level"]).startswith("noise")
    assert by_of["oracle_power_ridge"]["level"] == "descriptive"


def write_day(
    directory: Path, label: str, keys: list[tuple[int, str, int]], value: float, flat: bool = True
) -> None:
    rows = [f"{w},{a},{c}" for w, a, c in keys]
    (directory / f"answers-{label}.csv").write_text(
        "window,axis,channel,lag,target_mean,mean_last_hour,mean_fleet_ridge\n"
        + "".join(f"{r},,0,1.0,0.9\n" for r in rows)
    )
    local = (*FORECASTERS, *FLAT) if flat else FORECASTERS
    for kind, names, answer in (
        ("local", local, "0.8"),
        ("chronos2", RUNGS, str(value)),
        ("trees", tuple(TREES), "0.85"),
    ):
        (directory / f"{kind}-{label}.csv").write_text(
            "window,axis,channel,"
            + ",".join(f"mean_{n}" for n in names)
            + "\n"
            + "".join(f"{r}," + ",".join(answer for _ in names) + "\n" for r in rows)
        )


def test_the_first_month_is_read_on_every_start_and_the_curve_only_on_its_own(
    tmp_path: Path,
) -> None:
    (tmp_path / "starts-farm.csv").write_text(
        f"start,hour,curve\n{EARLY},0,1\n{LATE},0,0\n{date(2020, 9, 1)},0,1\n"
    )
    for number in (2, 7, 14):
        keys = cells_for([EARLY, LATE, date(2020, 9, 1)], hour=number * 24)
        write_day(tmp_path, f"farm-N{number}", keys, value=0.4)
    write_day(tmp_path, "farm-N30", cells_for([EARLY, date(2020, 9, 1)], hour=720), value=0.4)

    comparisons, losses = read(tmp_path, "farm", least_gain=0.8, resamples=100, seed=1)

    month = [r for r in comparisons if r["set"] == "first month"]
    assert {r["days"] for r in month} == {"2-14"}
    assert month[0]["starts"] == 3
    assert {r["set"] for r in comparisons if str(r["set"]).startswith("group")} == {
        f"group: {g}" for g, names in GROUPS.items() if {TARGETS[0], TARGETS[10]} & set(names)
    }
    assert {(r["days"], r["starts"]) for r in comparisons if r["set"] == "curve"} == {
        ("2", 2),
        ("7", 2),
        ("14", 2),
        ("30", 2),
    }
    assert {r["answer"] for r in losses} >= {"chronos2_47", "auto_ets_flat", "fleet_ridge"}
    groups = {r["level"] for r in comparisons if str(r["set"]).startswith("group")}
    assert groups == {"descriptive"}
    levels = {(r["of"], str(r["against"]).split(",")[0]): r["level"] for r in month}
    assert levels[("chronos2_47", "with the farm")] != "descriptive"
    assert levels[("chronos2_47", "without the farm")] == "descriptive"
    assert levels[("chronos2_nbm", "normal behaviour without the farm")] != "descriptive"
    assert levels[("chronos2_nbm", "normal behaviour with the farm")] != "descriptive"
    assert levels[("fleet_nbm_trees", "normal behaviour without the farm")] == "descriptive"
    assert levels[("chronos2_nbm_own", "normal behaviour with the farm")] == "descriptive"
    verdict = next(r for r in month if str(r["against"]).startswith("with the farm"))
    bar = 2 + len(FORECASTERS) + len(FLAT) + 1
    assert str(verdict["against"]).startswith(f"with the farm, best of {bar}: ")


def test_a_day_with_a_file_missing_is_refused(tmp_path: Path) -> None:
    for number in (2, 7, 14):
        write_day(tmp_path, f"farm-N{number}", cells_for([EARLY], hour=number * 24), value=0.4)
    (tmp_path / "local-farm-N7.csv").unlink()

    with pytest.raises(ValueError, match="not written"):
        read(tmp_path, "farm", least_gain=0.8, resamples=10, seed=1)


def test_a_local_file_without_the_flat_answers_is_refused(tmp_path: Path) -> None:
    for number in (2, 7, 14):
        keys = cells_for([EARLY], hour=number * 24)
        write_day(tmp_path, f"farm-N{number}", keys, 0.4, flat=False)

    with pytest.raises(ValueError, match="auto_ets_flat"):
        read(tmp_path, "farm", least_gain=0.8, resamples=10, seed=1)


def test_a_ridge_on_the_turbines_own_hours_missing_a_cell_is_left_out_with_its_share() -> None:
    keys = cells_for([EARLY, LATE])
    answers = {
        "chronos2_47": [0.5] * len(keys),
        "last_hour": [1.0] * len(keys),
        "ridge": [np.nan] + [0.9] * (len(keys) - 1),
    }

    (row,) = comparison_rows(
        day(keys, answers, target=0.1),
        VERDICTS[:1],
        reading="first month",
        levels=True,
        least_gain=0.8,
        resamples=50,
        seed=1,
    )

    assert row["against"] == "with the farm, best of 1: last_hour"
    assert row["cells"] == len(keys)
    assert row["left_out"] == f"ridge {1 - 1 / len(keys):.3f}"


def test_a_window_hit_exactly_is_refused_but_dropped_and_counted_where_asked() -> None:
    keys = cells_for([EARLY, LATE])
    assert keys[0][:2] == keys[1][:2]
    answers = {"chronos2_47": [0.5] * len(keys), "last_hour": [0.1, 0.1] + [1.0] * (len(keys) - 2)}

    def rows(drop: bool) -> list[dict[str, object]]:
        return comparison_rows(
            day(keys, answers, target=0.1),
            VERDICTS[:1],
            reading="group: x",
            levels=False,
            least_gain=0.8,
            resamples=50,
            seed=1,
            drop_exact=drop,
        )

    with pytest.raises(ValueError, match="missed by nothing"):
        rows(False)
    (row,) = rows(True)
    assert row["left_out"] == "1 windows hit exactly"
    assert row["windows"] == len({(k[1], k[0]) for k in keys})


def test_a_normal_behaviour_comparison_reads_only_the_cells_whose_hour_was_known() -> None:
    keys = cells_for([EARLY, LATE])
    known = [k % 3 != 0 for k in range(len(keys))]
    answers = {
        "chronos2_nbm": [0.5 if k else np.nan for k in known],
        "last_hour": [1.0] * len(keys),
        "nbm_ridge": [0.7 if k else np.nan for k in known],
        "local_nbm_arx": [0.6 if k else np.nan for k in known],
    }

    (row,) = comparison_rows(
        day(keys, answers, target=0.1),
        VERDICTS[2:3],
        reading="first month",
        levels=True,
        least_gain=0.8,
        resamples=50,
        seed=1,
    )

    assert row["cells"] == sum(known)
    assert row["against"] == "normal behaviour without the farm, best of 3: local_nbm_arx"
    assert row["left_out"] == ""


def test_the_bars_with_the_farm_hold_the_bars_without_it_and_no_forecaster() -> None:
    assert set(BARS["without the farm"]) < set(BARS["with the farm"])
    assert set(BARS["normal behaviour without the farm"]) < set(
        BARS["normal behaviour with the farm"]
    )
    assert set(BARS["normal behaviour with the farm"]) < set(
        BARS["normal behaviour with the farm and currents"]
    )
    every = {a for bar in BARS.values() for a in bar}
    assert not any(a.startswith("chronos2") or a.startswith("oracle") for a in every)
    assert not any("fleet" in a for a in BARS["normal behaviour without the farm"])


def test_an_answer_of_a_bar_missing_a_few_cells_keeps_its_place_and_those_cells_go() -> None:
    keys = cells_for([EARLY, LATE, date(2021, 1, 1)], hour=24) * 1
    keys = [(h + 100 * k, a, c) for k in range(4) for h, a, c in keys]
    gaps = {3, 40}
    answers = {
        "chronos2_47": [0.5] * len(keys),
        "last_hour": [1.0] * len(keys),
        "fleet_trees": [np.nan if k in gaps else 0.2 for k in range(len(keys))],
    }

    (row,) = comparison_rows(
        day(keys, answers, target=0.1),
        VERDICTS[:1],
        reading="first month",
        levels=True,
        least_gain=0.8,
        resamples=50,
        seed=1,
    )

    assert row["against"] == "with the farm, best of 2: fleet_trees"
    assert row["cells"] == len(keys) - len(gaps)
    assert row["left_out"] == f"{len(gaps) / len(keys):.3f} of the cells unanswered by the bar"


def test_the_cells_own_regression_is_the_bar_without_the_farm_where_it_leads() -> None:
    keys = cells_for([EARLY, LATE])
    answers = {
        "chronos2_nbm": [0.5] * len(keys),
        "last_hour": [1.0] * len(keys),
        "local_nbm_arx": [0.45] * len(keys),
    }

    (row,) = comparison_rows(
        day(keys, answers, target=0.1),
        VERDICTS[2:3],
        reading="first month",
        levels=True,
        least_gain=0.8,
        resamples=50,
        seed=1,
    )

    assert row["against"] == "normal behaviour without the farm, best of 2: local_nbm_arx"
    assert row["ratio"] == pytest.approx(0.16 / 0.1225)
