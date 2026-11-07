"""The pretext input report: history cells, forecasts, skills, the rule, the block, on CPU."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
from scripts.telemetry_pretext_input_report import (
    CROSS,
    FLOOR_SHARE,
    LEAST_READ_SKILL,
    LEAST_TEACHING_SKILL,
    LENGTHS,
    LOG_COUNT,
    MEAN,
    OWN,
    PENALTIES,
    PENALTY_COLUMNS,
    PRESENCE,
    READ,
    RESOLUTIONS,
    TARGETS,
    TEACH,
    Arm,
    Believable,
    Moments,
    MonthErrors,
    arms,
    autocorrelation,
    block_months,
    chosen_arm,
    chosen_penalty,
    clipped_hours,
    departures,
    enough_history,
    errors_of,
    floored_scale,
    forecast_of,
    gathered,
    history_cells,
    hour_summaries,
    level_free,
    main,
    median_of_hours,
    month_draws,
    month_errors,
    monthly_median_errors,
    pooled_rows,
    skills,
    standardised,
    write_rows,
    yearly,
)

FEATURES = BinnedTokens.N_FEATURES


def published_hours(levels: list[float | None]) -> np.ndarray:
    """``[hours, 6, 9]``: each hour read once in every ten minutes at its level, or empty."""
    tokens = []
    for level in levels:
        cells = np.zeros((6, FEATURES))
        if level is not None:
            cells[:, PRESENCE] = 1.0
            cells[:, LOG_COUNT] = math.log1p(1)
            cells[:, 2:7] = level
            cells[:, 7] = 0.5
        tokens.append(cells)
    return np.stack(tokens)


def test_an_hours_summary_is_its_six_cells_as_one() -> None:
    hours = published_hours([2.0, None])

    summary = hour_summaries(hours)

    assert summary.shape == (2, FEATURES)
    assert summary[0, PRESENCE] == 1.0
    assert summary[0, MEAN] == pytest.approx(2.0)
    assert summary[0, LOG_COUNT] == pytest.approx(math.log1p(6))
    assert summary[1, PRESENCE] == 0.0


def test_a_missing_window_is_cells_of_nothing() -> None:
    series = np.arange(6, dtype=np.float64).reshape(3, 2)

    out = gathered(series, np.asarray([[2, -1]], dtype=np.int64))

    assert out.tolist() == [[[4.0, 5.0], [0.0, 0.0]]]


def test_the_history_holds_the_hours_before_at_the_arms_resolution() -> None:
    channel = published_hours([1.0, 2.0, 3.0, 4.0])
    before = np.asarray([[0, 1, 2]], dtype=np.int64)

    hourly = history_cells(channel, before, Arm("60min", 3, OWN)).reshape(2, FEATURES)
    fine = history_cells(channel, before, Arm("10min", 3, OWN)).reshape(12, FEATURES)

    assert hourly[:, MEAN].tolist() == pytest.approx([2.0, 3.0])
    assert fine[:, MEAN].tolist() == pytest.approx([2.0] * 6 + [3.0] * 6)


def test_an_hour_with_no_reading_carries_the_last_one_forward() -> None:
    channel = published_hours([5.0, None])

    laid = history_cells(
        channel, np.asarray([[0, 1]], dtype=np.int64), Arm("60min", 3, OWN)
    ).reshape(2, FEATURES)

    assert laid[1, PRESENCE] == 0.0
    assert laid[1, MEAN] == pytest.approx(5.0)


def test_the_median_reads_only_the_hours_held() -> None:
    hours = hour_summaries(published_hours([1.0, None, 3.0, 10.0]))[None]

    median, held = median_of_hours(hours)

    assert held.tolist() == [3]
    assert median[0, 0] == pytest.approx(3.0)


def test_history_is_enough_from_half_the_hours_before() -> None:
    assert enough_history(np.asarray([2, 3], dtype=np.int64), 6).tolist() == [False, True]
    assert enough_history(np.asarray([23, 24], dtype=np.int64), 48).tolist() == [False, True]


def test_moments_added_are_the_moments_of_both_sets_of_rows() -> None:
    rng = np.random.default_rng(4)
    inputs, targets = rng.normal(size=(30, 3)), rng.normal(size=(30, 2))

    added = Moments.of(inputs[:10], targets[:10]) + Moments.of(inputs[10:], targets[10:])
    whole = Moments.of(inputs, targets)

    assert added.rows == 30
    assert added.gram == pytest.approx(whole.gram)
    assert added.cross == pytest.approx(whole.cross)


def test_the_ridge_forecasts_each_target_under_its_own_penalty() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    inputs = np.column_stack((np.ones(200), features))
    targets = np.column_stack((features @ [1.0, -2.0, 0.5] + 3.0, features[:, 0] + 3.0))
    every = np.arange(4)

    answers = forecast_of(Moments.of(inputs, targets), inputs, every, np.asarray([1e-6, 1e6]))

    assert answers[:, 0] == pytest.approx(targets[:, 0], abs=1e-3)
    assert np.abs(answers[:, 1]).max() < 1e-4


def test_the_largest_penalty_shrinks_a_forecast_to_the_medians_answer() -> None:
    rng = np.random.default_rng(6)
    inputs = np.column_stack((np.ones(300), rng.normal(size=(300, 4))))
    departure = rng.normal(size=(300, 1)) + 0.5

    shrunk = forecast_of(
        Moments.of(inputs, departure), inputs, np.arange(5), np.asarray([PENALTIES[-1]])
    )

    assert shrunk.tolist() == [[0.0]] * 300


def test_the_grid_ends_in_the_medians_answer_however_wide_the_window() -> None:
    rng = np.random.default_rng(7)
    walk = np.cumsum(rng.normal(size=(400, 1)), axis=0)
    inputs = np.column_stack((np.ones(400), walk + 0.01 * rng.normal(size=(400, 300))))
    departure = rng.normal(size=(400, 1))

    shrunk = forecast_of(
        Moments.of(inputs, departure), inputs, np.arange(301), np.asarray([PENALTIES[-1]])
    )

    assert np.abs(shrunk).max() == 0.0


def test_a_forecast_over_some_columns_reads_those_alone() -> None:
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(300, 2))
    targets = inputs[:, [1]] * 2.0

    own = forecast_of(
        Moments.of(inputs, targets), inputs, np.asarray([0], dtype=np.intp), np.asarray([1e-6])
    )

    assert np.corrcoef(own[:, 0], targets[:, 0])[0, 1] == pytest.approx(0.0, abs=0.2)


def test_the_penalty_is_small_where_the_inputs_tell_and_large_where_they_do_not() -> None:
    rng = np.random.default_rng(1)
    inputs = rng.normal(size=(400, 5))
    targets = np.column_stack((inputs @ np.arange(1.0, 6.0), rng.normal(size=400)))
    inner = np.arange(400) < 300

    chosen = chosen_penalty(
        Moments.of(inputs[inner], targets[inner]), inputs[~inner], targets[~inner], np.arange(5)
    )

    assert chosen[0] < 0.01
    assert chosen[1] >= 1.0


def test_columns_without_spread_are_dropped_and_named() -> None:
    values = np.column_stack((np.arange(4.0), np.ones(4), np.arange(4.0) * 2))

    scaled, kept = standardised(values, np.ones(4, dtype=np.bool_))

    assert kept.tolist() == [True, False, True]
    assert scaled.mean(axis=0) == pytest.approx([0.0, 0.0])
    assert scaled.std(axis=0) == pytest.approx([1.0, 1.0])


def test_a_level_shared_by_the_history_and_its_median_leaves_no_trace() -> None:
    channel = published_hours([None, 1.0, 1.5, 0.5, 2.0])
    before = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64)
    arm = Arm("30min", 6, OWN)
    cells = history_cells(channel, before, arm).reshape(1, arm.history_bins, FEATURES)
    shifted = cells.copy()
    shifted[..., 2:7] += 40.0

    plain = level_free(cells, np.asarray([[1.25, 1.9]]))
    drifted = level_free(shifted, np.asarray([[41.25, 1.9]]))

    assert drifted == pytest.approx(plain)
    assert plain[0, :2, 2:7].tolist() == [[0.0] * 5] * 2
    assert plain[0, 2, MEAN] == pytest.approx(-0.25)
    assert plain[0, 2, LOG_COUNT] == pytest.approx(math.log1p(3) - 1.9)


def test_a_bin_without_a_reading_holds_no_count_after_the_first_reading_either() -> None:
    channel = published_hours([1.0, None])
    arm = Arm("60min", 3, OWN)
    cells = history_cells(channel, np.asarray([[0, 1]], dtype=np.int64), arm)

    plain = level_free(cells.reshape(1, 2, FEATURES), np.asarray([[1.0, 1.9]]))

    assert plain[0, 1, LOG_COUNT] == 0.0
    assert plain[0, 1, MEAN] == pytest.approx(0.0)


def test_another_channels_last_hour_reads_as_its_departure_or_nothing() -> None:
    last = np.zeros((2, 2, FEATURES))
    last[:, 0, PRESENCE] = 1.0
    last[:, 0, list(TARGETS)] = 5.0
    median = np.full((2, 2, 5), 4.0)
    median[1, 0] = np.nan

    away = departures(last, median)

    assert away[0, 0].tolist() == [1.0] * 5 + [1.0]
    assert away[1, 0].tolist() == [0.0] * 6
    assert away[:, 1].tolist() == [[0.0] * 6] * 2


def test_a_scale_is_floored_where_the_median_is_nearly_exact() -> None:
    targets = np.column_stack((np.arange(10.0), np.full(10, 3.0)))

    scale = floored_scale(np.asarray([1e-9, 0.0]), targets)

    assert scale[0] == pytest.approx(FLOOR_SHARE * 2.5)
    assert scale[1] == 0.0
    assert floored_scale(np.asarray([4.0, 0.0]), targets)[0] == pytest.approx(4.0)


def test_a_measure_that_loses_on_its_own_rows_is_not_believable() -> None:
    assert Believable(least_teaching=0.0, least_read=-0.5).holds
    assert not Believable(least_teaching=LEAST_TEACHING_SKILL - 0.01, least_read=0.1).holds
    assert not Believable(least_teaching=0.0, least_read=LEAST_READ_SKILL - 0.01).holds


def test_a_reading_written_before_the_correction_is_refused(tmp_path: Path) -> None:
    write_rows(
        tmp_path / "errors.csv", ["resolution", "month"], [{"resolution": "60min", "month": "a"}]
    )

    with pytest.raises(SystemExit, match="holds no side"):
        main(["read", "--dir", str(tmp_path)])


def test_errors_are_scaled_and_summed_per_month() -> None:
    targets = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    forecast = targets + np.asarray([[1.0, 5.0], [2.0, 5.0], [0.0, 5.0]])
    median = targets + 4.0
    scale = np.asarray([2.0, 0.0])

    each = month_errors(
        forecast,
        median,
        targets,
        scale,
        np.asarray([True, True, False], dtype=np.bool_),
        ["a", "a", "b"],
    )

    assert each.forecast == {"a": pytest.approx(1.5)}
    assert each.median == {"a": pytest.approx(4.0)}
    assert each.cells == {"a": 2}


def test_a_daily_cycle_correlates_at_a_day_and_against_at_half_a_day() -> None:
    hours = np.arange(24 * 40, dtype=np.float64)
    series = np.sin(2 * np.pi * hours / 24)
    series[::7] = np.nan

    acf = autocorrelation(series, 24)

    assert acf[23] == pytest.approx(1.0, abs=0.05)
    assert acf[11] == pytest.approx(-1.0, abs=0.05)


def test_a_flat_series_has_no_autocorrelation() -> None:
    assert np.isnan(autocorrelation(np.ones(50), 3)).all()


def test_errors_repeating_after_a_year_are_yearly() -> None:
    months = np.arange(48, dtype=np.float64)
    cycle = {c: np.sin(2 * np.pi * months / 12) + 0.01 * c for c in range(5)}
    rng = np.random.default_rng(2)
    noise = {c: rng.normal(size=48) for c in range(5)}

    assert yearly(cycle).present
    assert not yearly(noise).present
    assert yearly(cycle).band == pytest.approx(1.96 / math.sqrt(48))


def test_fewer_than_three_years_of_months_are_not_asked() -> None:
    short = yearly({0: np.sin(np.arange(21.0))})

    assert not short.read
    assert not short.present


def test_a_block_takes_three_months_or_the_hours_a_scale_needs() -> None:
    assert block_months([730] * 84, yearly_present=False) == 3
    assert block_months([730] * 84, yearly_present=True) == 12
    assert block_months([730] * 10 + [400] * 6, yearly_present=False) == 5


def test_a_block_is_refused_where_the_months_cannot_hold_one() -> None:
    with pytest.raises(ValueError, match="short of a block"):
        block_months([500, 500, 500], yearly_present=False)
    with pytest.raises(ValueError, match="no month to pretrain on"):
        block_months([1000] * 3, yearly_present=False)


def errors_where(skill: dict[Arm, float], months: int, seed: int = 3) -> dict[Arm, MonthErrors]:
    """Per-month errors whose skill over the median is about ``skill``, with a little noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for arm in arms():
        level = skill.get(arm, 0.0)
        noise = rng.normal(0.0, 0.002, size=months)
        forecast = {f"m{m:02d}": 100.0 * (1.0 - level + noise[m]) for m in range(months)}
        median = {f"m{m:02d}": 100.0 for m in range(months)}
        out[arm] = MonthErrors(forecast, median, dict.fromkeys(median, 10))
    return out


def ruled(levels: dict[tuple[str, int], float], months: int = 28) -> Arm:
    skill = {Arm(r, length, CROSS): v for (r, length), v in levels.items()}
    errors = errors_where(skill, months)
    names = sorted(next(iter(errors.values())).median)
    _, drawn = skills(errors, month_draws(len(names), 2000, 1), names)
    return chosen_arm(drawn)


def test_the_rule_takes_the_shortest_length_no_longer_one_beats() -> None:
    levels = {("60min", 6): 0.10, ("60min", 12): 0.20, ("60min", 24): 0.30, ("60min", 48): 0.30}

    assert ruled(levels) == Arm("60min", 24, CROSS)


def test_the_rule_takes_finer_cells_only_where_they_forecast_better() -> None:
    same = {(r, length): 0.2 for r in RESOLUTIONS for length in LENGTHS}
    finer = {**same, **{("10min", length): 0.3 for length in LENGTHS}}

    assert ruled(same) == Arm("60min", 6, CROSS)
    assert ruled(finer) == Arm("10min", 6, CROSS)


def test_skills_read_one_less_the_forecasts_share_of_the_medians_error() -> None:
    arm = Arm("60min", 6, OWN)
    errors = {arm: MonthErrors({"a": 30.0, "b": 50.0}, {"a": 100.0, "b": 100.0}, {})}

    found, drawn = skills(errors, month_draws(2, 500, 1), ["a", "b"])

    assert found[arm].skill == pytest.approx(0.6)
    assert found[arm].low <= 0.6 <= found[arm].high
    assert drawn[arm].shape == (500,)


def test_every_arm_names_its_cells() -> None:
    assert len(arms()) == len(RESOLUTIONS) * len(LENGTHS) * 2
    assert Arm("10min", 48, OWN).tokens == 288
    assert Arm("30min", 6, OWN).history_bins == 10
    assert TARGETS[-1] == LOG_COUNT


def test_the_reading_writes_the_choice_from_what_was_measured(tmp_path: Path) -> None:
    levels = {Arm(r, length, CROSS): 0.2 for r in RESOLUTIONS for length in LENGTHS}
    rows = [
        {
            "side": side,
            "resolution": arm.resolution,
            "length": arm.length,
            "variant": arm.variant,
            "month": month,
            "forecast": forecast,
            "median": each.median[month],
            "cells": each.cells[month],
            "channel": 7,
        }
        for side in (TEACH, READ)
        for arm, each in errors_where(levels, 28).items()
        for month, forecast in each.forecast.items()
    ]
    write_rows(tmp_path / "errors.csv", list(rows[0]), rows)
    monthly = [{"channel": 7, "month": f"m{m:02d}", "median_error": 1.0} for m in range(28)]
    write_rows(tmp_path / "monthly.csv", ["channel", "month", "median_error"], monthly)
    windows = [{"month": f"m{m:02d}", "windows": 730} for m in range(28)]
    write_rows(tmp_path / "windows.csv", ["month", "windows"], windows)
    shares = [
        {
            "resolution": "60min",
            "length": 6,
            "variant": CROSS,
            "channel": 7,
            "target": "mean",
            "share": share,
        }
        for share in (PENALTIES[0], 0.01, 0.1, PENALTIES[-1])
    ]
    write_rows(tmp_path / "penalties.csv", PENALTY_COLUMNS, shares)

    main(["read", "--dir", str(tmp_path)])

    choice = json.loads((tmp_path / "choice.json").read_text())
    assert (choice["resolution"], choice["length"]) == ("60min", 6)
    assert choice["block_months"] == 3
    assert choice["yearly_read"] is False
    assert choice["penalties_at_the_least"] == pytest.approx(0.25)
    assert choice["penalties_at_the_most"] == pytest.approx(0.25)
    assert choice["believable"] is True
    assert len(errors_of([{k: str(v) for k, v in r.items()} for r in rows])) == len(arms())


def test_a_target_that_never_moved_leaves_the_monthly_error_of_the_others() -> None:
    targets = np.asarray([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    median = np.asarray([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    rows = np.asarray([True, True, False], dtype=np.bool_)

    found = monthly_median_errors(
        median, targets, np.asarray([1.0, 0.0]), rows, ["a", "a", "b"], ["a", "b"]
    )

    assert found[0][0] == "a"
    assert found[0][1] == pytest.approx(1.0)
    assert found[1][0] == "b"
    assert math.isnan(found[1][1])


def test_the_bars_hours_are_clipped_where_the_targets_are() -> None:
    hours = np.zeros((1001, FEATURES))
    hours[:, MEAN] = np.arange(1001.0)
    hours[0, MEAN] = -1e9

    clipped = clipped_hours(hours, hours[1:])

    assert clipped[0, MEAN] == pytest.approx(np.quantile(np.arange(1.0, 1001.0), 0.001))
    assert clipped[:, PRESENCE].tolist() == hours[:, PRESENCE].tolist()


def test_a_month_of_one_mission_is_not_the_same_month_of_another() -> None:
    row = {"side": READ, "month": "2001-01", "forecast": "1.0"}

    pooled = pooled_rows({"m1": [row], "m2": [row]})

    assert [r["month"] for r in pooled] == ["m1:2001-01", "m2:2001-01"]
    assert row["month"] == "2001-01"


def test_the_pooled_rule_reads_every_run_and_names_its_months(tmp_path: Path) -> None:
    levels = {Arm(r, length, CROSS): 0.2 for r in RESOLUTIONS for length in LENGTHS}
    for run in ("a", "b"):
        rows = [
            {
                "side": side,
                "resolution": arm.resolution,
                "length": arm.length,
                "variant": arm.variant,
                "month": month,
                "forecast": forecast,
                "median": each.median[month],
                "cells": each.cells[month],
                "channel": 7,
            }
            for side in (TEACH, READ)
            for arm, each in errors_where(levels, 10).items()
            for month, forecast in each.forecast.items()
        ]
        (tmp_path / run).mkdir()
        write_rows(tmp_path / run / "errors.csv", list(rows[0]), rows)
    out = tmp_path / "pooled.json"

    main(
        [
            "pool",
            "--dirs",
            str(tmp_path / "a"),
            str(tmp_path / "b"),
            "--names",
            "a",
            "b",
            "--out",
            str(out),
        ]
    )

    choice = json.loads(out.read_text())
    assert choice["months"] == 20
    assert (choice["resolution"], choice["length"]) == ("60min", 6)
