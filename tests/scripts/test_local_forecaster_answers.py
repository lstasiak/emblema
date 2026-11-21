"""The local forecasters' answers: lags read, hours filled, cells grouped by season and written."""

import csv
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.chronos2_answers import ContextRow
from scripts.history_forecast_reading import read_published
from scripts.local_forecaster_answers import (
    COLUMNS,
    FLAT,
    FORECASTERS,
    answer,
    by_season,
    filled,
    read_lags,
    season_of,
    write_answers,
)


def row(window: int, channel: int, means: list[float], axis: str = "M1") -> ContextRow:
    return ContextRow(window, axis, channel, np.array(means, dtype=np.float32))


def last_hour(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
    """Forecasters standing in: each answers the last hour plus its position and the season."""
    return {name: series[:, -1] + k + 10 * season for k, name in enumerate(FORECASTERS)}


def test_a_blank_lag_reads_as_one(tmp_path: Path) -> None:
    path = tmp_path / "answers.csv"
    path.write_text("window,axis,channel,lag,target_mean\n4,M1,2,18,0.1\n4,M1,3,,0.2\n")

    assert read_lags(path) == {(4, "M1", 2): 18, (4, "M1", 3): 1}


def test_a_missing_hour_carries_the_one_before_and_the_first_reading_runs_back() -> None:
    means = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan], dtype=np.float32)

    assert filled(means).tolist() == [1.0, 1.0, 1.0, 1.0, 3.0, 3.0]


def test_a_context_with_no_reading_cannot_be_filled() -> None:
    with pytest.raises(ValueError, match="no hour read"):
        filled(np.array([np.nan, np.nan], dtype=np.float32))


@pytest.mark.parametrize(("lag", "season"), [(1, 1), (2, 2), (18, 18), (23, 23), (24, 1), (0, 1)])
def test_a_season_is_the_period_only_when_it_fits_twice_in_the_context(
    lag: int, season: int
) -> None:
    assert season_of(lag, 47) == season


def test_a_cell_without_an_answer_row_is_refused() -> None:
    with pytest.raises(ValueError, match="window 7 channel 2"):
        by_season([row(7, 2, [0.0] * 47)], {})


def test_every_cell_gets_each_forecasters_answer_under_its_own_season() -> None:
    rows = [row(1, 2, [0.0] * 46 + [5.0]), row(1, 3, [0.0] * 46 + [7.0])]
    lags = {(1, "M1", 2): 18, (1, "M1", 3): 1}

    answered = answer(rows, lags, last_hour)

    assert answered[(1, "M1", 2)] == {
        "auto_ets": 185.0,
        "auto_arima": 186.0,
        "auto_theta": 187.0,
        "seasonal_naive": 188.0,
    }
    assert answered[(1, "M1", 3)]["auto_ets"] == 17.0


def test_a_fit_answering_other_cells_or_forecasters_is_refused() -> None:
    rows = [row(1, 2, [0.0] * 47), row(1, 3, [1.0] * 47)]
    lags = {(1, "M1", 2): 1, (1, "M1", 3): 1}

    def short(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
        return dict.fromkeys(FORECASTERS, series[:1, -1])

    def partial(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
        return {"auto_ets": series[:, -1]}

    with pytest.raises(ValueError, match="answered 1 cells for 2"):
        answer(rows, lags, short)
    with pytest.raises(ValueError, match="the fit answered"):
        answer(rows, lags, partial)


def first_hour(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
    """Forecasters standing in that answer each series by its first hour."""
    return dict.fromkeys(FORECASTERS, series[:, 0])


def test_the_answers_go_back_to_their_own_cells_within_a_season() -> None:
    rows = [row(w, c, [float(10 * w + c)] * 47) for w in (1, 2) for c in (3, 4)]
    lags = {(r.window, r.axis, r.channel): 18 for r in rows}

    answered = answer(rows, lags, first_hour)

    assert {key: values["auto_ets"] for key, values in answered.items()} == {
        (1, "M1", 3): 13.0,
        (1, "M1", 4): 14.0,
        (2, "M1", 3): 23.0,
        (2, "M1", 4): 24.0,
    }


def test_an_answer_that_is_not_finite_is_refused() -> None:
    rows = [row(1, 2, [0.0] * 47)]

    def broken(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
        return dict.fromkeys(FORECASTERS, np.full(len(series), np.nan))

    with pytest.raises(ValueError, match="not finite at season 1"):
        answer(rows, {(1, "M1", 2): 1}, broken)


def test_the_answers_are_written_in_the_context_order_and_read_back_by_the_reading(
    tmp_path: Path,
) -> None:
    rows = [row(2, 5, [1.0] * 47), row(1, 3, [2.0] * 47)]
    answered = {
        (2, "M1", 5): {n: 0.5 + k for k, n in enumerate(FORECASTERS)},
        (1, "M1", 3): dict.fromkeys(FORECASTERS, 0.25),
    }
    path = tmp_path / "local.csv"

    write_answers(path, rows, answered)

    with path.open(newline="") as handle:
        read = list(csv.reader(handle))
    assert tuple(read[0]) == COLUMNS
    assert [line[:3] for line in read[1:]] == [["2", "M1", "5"], ["1", "M1", "3"]]
    assert read_published(path) == {
        n: {(2, "M1", 5): 0.5 + k, (1, "M1", 3): 0.25} for k, n in enumerate(FORECASTERS)
    }


def test_with_flat_the_seasonal_forecasters_answer_again_with_no_season_and_are_written(
    tmp_path: Path,
) -> None:
    rows = [row(1, 2, [0.0] * 46 + [5.0]), row(1, 3, [0.0] * 46 + [7.0])]
    lags = {(1, "M1", 2): 18, (1, "M1", 3): 1}
    seasons: list[int] = []

    def counted(series: NDArray[np.float64], season: int) -> Mapping[str, NDArray[np.float64]]:
        seasons.append(season)
        return last_hour(series, season)

    answered = answer(rows, lags, counted, flat=True)
    path = tmp_path / "local.csv"
    write_answers(path, rows, answered, flat=True)

    assert sorted(seasons) == [1, 1, 18]
    assert answered[(1, "M1", 2)] == {
        "auto_ets": 185.0,
        "auto_arima": 186.0,
        "auto_theta": 187.0,
        "seasonal_naive": 188.0,
        "auto_ets_flat": 15.0,
        "auto_arima_flat": 16.0,
        "auto_theta_flat": 17.0,
    }
    assert answered[(1, "M1", 3)]["auto_ets_flat"] == answered[(1, "M1", 3)]["auto_ets"]
    assert read_published(path)["auto_theta_flat"] == {(1, "M1", 2): 17.0, (1, "M1", 3): 19.0}
    assert set(FLAT) <= set(read_published(path))
