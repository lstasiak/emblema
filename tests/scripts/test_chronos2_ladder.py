"""The published forecaster's rungs: contexts before the hour, library series laid against it."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.chronos2_ladder import (
    CONTEXT,
    RUNGS,
    Selection,
    answered,
    before,
    dense,
    laid,
    rung_inputs,
)

AXIS = "M"


def test_hours_before_an_hour_are_each_channels_padded_before_the_first() -> None:
    values = dense(np.array([0, 1, 3]), np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]))

    context = before(values, 3, 5)

    assert context.shape == (2, 5)
    assert context[0].tolist() == pytest.approx([np.nan, np.nan, 1.0, 2.0, np.nan], nan_ok=True)


def test_a_library_series_hands_the_hours_it_was_chosen_on_and_the_hour_it_meets() -> None:
    series = np.arange(100.0)

    past, future = laid(series, 60, CONTEXT + 3)

    assert np.isnan(past[:3]).all()
    assert past[3:].tolist() == list(np.arange(60.0 - CONTEXT, 60.0))
    assert future.tolist() == [60.0]
    short, _ = laid(series, 60, 4)
    assert short.tolist() == [56.0, 57.0, 58.0, 59.0]
    early, _ = laid(series, 2, 4)
    assert np.isnan(early[:2]).all()
    assert early[2:].tolist() == [0.0, 1.0]


def selections_for(cells: Sequence[tuple[int, int]]) -> dict[str, dict[Any, Selection]]:
    """Every rung's series for these (window, channel) cells: library channel 0 meeting hour 60."""
    return {
        rung: {(w, AXIS, c): Selection(0, 60) for w, c in cells}
        for rung in ("unrelated", "kind", "analog")
    }


def test_a_rung_hands_the_target_and_one_covariate_per_covered_channel() -> None:
    values = np.ones((200, 3))
    library = np.arange(100.0)[:, None]

    handed, covered = rung_inputs(
        "chronos2_analog", 150, AXIS, values, library, selections_for([(150, 2)]), cap=64
    )

    assert covered == (2,)
    assert handed["target"].shape == (3, CONTEXT)
    assert handed["past_covariates"]["c2"].tolist() == list(np.arange(60.0 - CONTEXT, 60.0))
    assert handed["future_covariates"]["c2"].tolist() == [60.0]
    short, _ = rung_inputs("chronos2_own", 20, AXIS, values, library, {}, cap=64)
    assert short.shape == (3, 20)
    long, _ = rung_inputs("chronos2_own", 150, AXIS, values, library, {}, cap=64)
    assert long.shape == (3, 64)
    plain, _ = rung_inputs("chronos2_47", 150, AXIS, values, library, {}, cap=64)
    assert plain.shape == (3, CONTEXT)


def last_hour(batch: Sequence[Any]) -> list[NDArray[np.float32]]:
    """A model standing in: each variate's median is its last hour, plus one per covariate."""
    out = []
    for item in batch:
        target = item["target"] if isinstance(item, dict) else item
        shift = len(item["past_covariates"]) if isinstance(item, dict) else 0
        medians = np.nan_to_num(target[:, -1]) + shift
        out.append(np.repeat(medians[:, None, None], 3, axis=1).astype(np.float32))
    return out


def test_each_rung_answers_the_cells_it_covers_and_the_own_rungs_every_cell() -> None:
    values = np.tile(np.arange(200.0)[:, None], (1, 3))
    library = np.zeros((100, 1))
    cells = [(150, AXIS, 0), (150, AXIS, 2), (160, AXIS, 1)]

    out = answered(cells, values, library, selections_for([(150, 2)]), last_hour, median=1, cap=64)

    assert set(out) == set(RUNGS)
    assert out["chronos2_47"] == {key: float(key[0] - 1) for key in cells}
    assert out["chronos2_47_each"] == out["chronos2_47"]
    assert out["chronos2_analog"] == {(150, AXIS, 2): 150.0}
    assert out["chronos2_kind"] == {(150, AXIS, 2): 150.0}
    assert out["chronos2_oracle"] == {(150, AXIS, 2): 150.0}


def test_the_oracle_hands_the_target_channel_itself_and_the_forecast_hour() -> None:
    values = np.tile(np.arange(200.0)[:, None], (1, 3))

    handed, covered = rung_inputs(
        "chronos2_oracle", 150, AXIS, values, np.zeros((10, 1)), selections_for([(150, 1)]), cap=64
    )

    assert covered == (1,)
    assert handed["future_covariates"]["c1"].tolist() == [150.0]
    assert handed["past_covariates"]["c1"][-1] == 149.0


def test_a_model_answering_other_variates_is_refused() -> None:
    values = np.ones((200, 2))

    def one(batch: Sequence[Any]) -> list[NDArray[np.float32]]:
        return [np.zeros((1, 3, 1), dtype=np.float32) for _ in batch]

    with pytest.raises(ValueError, match="answered 1 variates for 2"):
        answered([(150, AXIS, 0)], values, np.zeros((10, 1)), {}, one, median=1, cap=64)
