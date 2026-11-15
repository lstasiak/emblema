"""The published forecaster's answers: the context read, grouped, forecast and written."""

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.chronos2_answers import (
    COLUMNS,
    ContextRow,
    forecast,
    grouped,
    median_index,
    read_context,
    write_forecasts,
)

QUANTILES = (0.1, 0.5, 0.9)


def context_file(path: Path, lines: Sequence[str]) -> Path:
    path.write_text("window,axis,channel,h01,h02,h03\n" + "\n".join(lines) + "\n")
    return path


def mean_of_finite(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
    """A model standing in: every quantile is the mean of the finite hours of each variate."""
    answers = []
    for series in batch:
        means = np.array([np.nanmean(variate) for variate in series], dtype=np.float32)
        answers.append(np.repeat(means[:, None, None], len(QUANTILES), axis=1))
    return answers


def test_the_context_is_read_with_blanks_as_missing_hours(tmp_path: Path) -> None:
    path = context_file(tmp_path / "context.csv", ["4,M1,2,0.5,,1.25", "4,M1,3,,2,"])

    rows = read_context(path)

    assert [(r.window, r.axis, r.channel) for r in rows] == [(4, "M1", 2), (4, "M1", 3)]
    assert rows[0].means == pytest.approx([0.5, np.nan, 1.25], nan_ok=True)
    assert np.isnan(rows[1].means[[0, 2]]).all()


def test_a_cell_with_no_finite_hour_is_refused(tmp_path: Path) -> None:
    path = context_file(tmp_path / "context.csv", ["4,M1,2,,,"])

    with pytest.raises(ValueError, match="window 4 channel 2"):
        read_context(path)


def rows_of(*cells: tuple[int, str, int]) -> list[ContextRow]:
    return [
        ContextRow(w, a, c, np.array([float(c), np.nan, 2.0 * c], dtype=np.float32))
        for w, a, c in cells
    ]


def test_a_windows_channels_are_handed_together() -> None:
    rows = rows_of((1, "M1", 7), (1, "M1", 8), (2, "M2", 9))

    assert [[r.channel for r in g] for g in grouped(rows)] == [[7, 8], [9]]


def test_the_median_is_the_quantile_kept() -> None:
    assert median_index(QUANTILES) == 1
    with pytest.raises(ValueError, match="no median"):
        median_index((0.1, 0.9))


def test_each_cell_gets_the_models_median_for_its_hour() -> None:
    rows = rows_of((1, "M1", 7), (1, "M1", 8), (2, "M2", 9))
    said: list[str] = []

    answered = forecast(grouped(rows), mean_of_finite, QUANTILES, batch=1, say=said.append)

    assert [r.channel for r, _ in answered] == [7, 8, 9]
    assert [m for _, m in answered] == pytest.approx([10.5, 12.0, 13.5])
    assert len(said) == 2
    assert said[-1].startswith("3 cells in")


def test_a_model_answering_another_number_of_variates_is_refused() -> None:
    rows = rows_of((1, "M1", 7), (1, "M1", 8))

    def one_short(batch: Sequence[NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        return [np.zeros((1, 3, 1), dtype=np.float32) for _ in batch]

    with pytest.raises(ValueError, match="1 variates for 2"):
        forecast(grouped(rows), one_short, QUANTILES)


def test_the_forecasts_are_written_one_row_a_cell(tmp_path: Path) -> None:
    (row,) = rows_of((4, "M1", 2))

    write_forecasts(tmp_path / "out.csv", [(row, 1.234567)])

    with (tmp_path / "out.csv").open(newline="") as handle:
        lines = list(csv.reader(handle))
    assert lines == [list(COLUMNS), ["4", "M1", "2", "1.23457"]]
