import math

import pytest

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError, UnknownKnobError
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting


def test_the_standard_setting_drops_nothing_and_reads_the_raw_readings() -> None:
    standard = EncoderSetting.standard()

    assert standard.dropout == 0.0
    assert standard.grid_resolution is None


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5, math.nan, math.inf])
def test_a_dropout_outside_zero_to_one_is_refused(dropout: float) -> None:
    with pytest.raises(InvalidEncoderSettingError, match="dropout"):
        EncoderSetting(dropout=dropout)


@pytest.mark.parametrize("resolution", [0.0, -1.0, math.nan, math.inf])
def test_a_grid_without_a_positive_finite_resolution_is_refused(resolution: float) -> None:
    with pytest.raises(InvalidEncoderSettingError, match="grid_resolution"):
        EncoderSetting(grid_resolution=resolution)


def test_both_knobs_turn_from_the_standard_setting() -> None:
    dropped = EncoderSetting.standard().tuned("dropout", "0.2")
    gridded = EncoderSetting.standard().tuned("grid_resolution", "1")

    assert dropped == EncoderSetting(dropout=0.2)
    assert gridded == EncoderSetting(grid_resolution=1.0)
    assert dropped != EncoderSetting.standard()


@pytest.mark.parametrize(
    ("knob", "value"),
    [("width", "64"), ("dropout", "a lot"), ("dropout", "1"), ("grid_resolution", "0")],
)
def test_a_knob_the_encoder_has_not_or_a_value_it_cannot_take_is_refused(
    knob: str, value: str
) -> None:
    with pytest.raises(UnknownKnobError):
        EncoderSetting.standard().tuned(knob, value)


def test_a_grid_lays_a_window_on_steps_by_the_rule_of_every_other_grid() -> None:
    assert EncoderSetting.standard().steps_over(48.0) is None
    assert EncoderSetting(grid_resolution=1.0).steps_over(48.0) == 48
    assert EncoderSetting(grid_resolution=2.0).steps_over(48.25) == 97


def test_a_run_records_both_columns_whatever_was_turned() -> None:
    assert EncoderSetting.standard().parameters() == {
        "encoder_dropout": 0.0,
        "grid_resolution": 0.0,
    }
    assert EncoderSetting(dropout=0.2, grid_resolution=1.0).parameters() == {
        "encoder_dropout": 0.2,
        "grid_resolution": 1.0,
    }


def test_a_description_names_only_what_was_turned_away_from_the_standard() -> None:
    assert EncoderSetting.standard().turned_away() == {}
    assert EncoderSetting(dropout=0.2).turned_away() == {"encoder_dropout": 0.2}
    assert EncoderSetting(grid_resolution=1.0).turned_away() == {"grid_resolution": 1.0}
