import math

import pytest

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError, UnknownKnobError
from emblema.evaluation.domain.transfer.encoder_setting import (
    CellSet,
    EncoderSetting,
    SummaryReading,
    TimeEncoding,
    ValueEmbedding,
)
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape
from emblema.evaluation.domain.transfer.layer_reading import LayerCombination, LayerReading


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
    [
        ("depth", "2"),
        ("dropout", "a lot"),
        ("dropout", "1"),
        ("grid_resolution", "0"),
        ("width", "0"),
        ("heads", "a few"),
        ("value_embedding", "cubic"),
        ("layer", "-1"),
        ("layer", "first"),
        ("summaries", "all"),
    ],
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


def test_a_run_records_every_column_whatever_was_turned() -> None:
    assert EncoderSetting.standard().parameters() == {
        "encoder_dropout": 0.0,
        "grid_resolution": 0.0,
        "bins": 0,
        "cells": "window",
        "value_embedding": "linear",
        "time_encoding": "fourier",
        "encoder_width": 0,
        "encoder_heads": 0,
        "encoder_layers": 0,
        "encoder_feedforward_width": 0,
        "value_clip": 0.0,
        "encoder_layer": "last",
        "summaries": "none",
    }
    turned = EncoderSetting(
        dropout=0.2,
        bins=12,
        cells=CellSet.CORPUS,
        value_embedding=ValueEmbedding.NONLINEAR,
        time_encoding=TimeEncoding.LEARNED,
        width=64,
        heads=16,
        layers=2,
        feedforward_width=128,
        layer=LayerReading(layer=3),
        summaries=SummaryReading.BESIDE,
    )
    assert turned.parameters() == {
        "encoder_dropout": 0.2,
        "grid_resolution": 0.0,
        "bins": 12,
        "cells": "corpus",
        "value_embedding": "nonlinear",
        "time_encoding": "learned",
        "encoder_width": 64,
        "encoder_heads": 16,
        "encoder_layers": 2,
        "encoder_feedforward_width": 128,
        "value_clip": 0.0,
        "encoder_layer": "3",
        "summaries": "beside",
    }


def test_a_description_names_only_what_was_turned_away_from_the_standard() -> None:
    assert EncoderSetting.standard().turned_away() == {}
    assert EncoderSetting(dropout=0.2).turned_away() == {"encoder_dropout": 0.2}
    assert EncoderSetting(grid_resolution=1.0).turned_away() == {"grid_resolution": 1.0}


def test_the_value_embedding_and_the_shape_are_named_only_when_turned() -> None:
    nonlinear = EncoderSetting.standard().tuned("value_embedding", "nonlinear")

    assert nonlinear.turned_away() == {"value_embedding": "nonlinear"}
    assert EncoderSetting(width=64).turned_away() == {"encoder_width": 64}
    assert EncoderSetting.standard().builds_its_own_encoder is False
    assert nonlinear.builds_its_own_encoder is True


def test_the_shape_is_whole_once_its_four_counts_are_turned_one_at_a_time() -> None:
    setting = EncoderSetting.standard()
    for knob, value in (("feedforward_width", "128"), ("heads", "16"), ("layers", "2")):
        setting = setting.tuned(knob, value)
        assert setting.shape is None
        assert setting.shape_partly_stated
        assert setting.builds_its_own_encoder

    whole = setting.tuned("width", "64")

    assert whole.shape == EncoderShape(width=64, heads=16, layers=2, feedforward_width=128)
    assert not whole.shape_partly_stated
    assert EncoderSetting.standard().shape is None
    assert not EncoderSetting.standard().shape_partly_stated


def test_a_whole_shape_whose_width_its_heads_cannot_split_is_refused() -> None:
    with pytest.raises(InvalidEncoderSettingError, match="multiple of heads"):
        _ = EncoderSetting(width=64, heads=5, layers=2, feedforward_width=128).shape


def test_a_clip_bounds_the_values_fed_and_changes_no_build() -> None:
    clipped = EncoderSetting.standard().tuned("value_clip", "5")

    assert clipped.value_clip == 5.0
    assert clipped.turned_away() == {"value_clip": 5.0}
    assert clipped.builds_its_own_encoder is False
    assert clipped.shape is None


@pytest.mark.parametrize("bound", ["0", "-5", "inf", "nan", "five"])
def test_a_clip_that_is_not_a_positive_finite_number_is_refused(bound: str) -> None:
    with pytest.raises(UnknownKnobError):
        EncoderSetting.standard().tuned("value_clip", bound)


@pytest.mark.parametrize(
    ("value", "reading"),
    [
        ("0", LayerReading(layer=0)),
        ("4", LayerReading(layer=4)),
        ("mean", LayerReading(combination=LayerCombination.MEAN)),
        ("concat", LayerReading(combination=LayerCombination.CONCATENATION)),
    ],
)
def test_the_layer_read_turns_and_is_named_only_when_turned(
    value: str, reading: LayerReading
) -> None:
    turned = EncoderSetting.standard().tuned("layer", value)

    assert turned.layer == reading
    assert turned.turned_away() == {"encoder_layer": value}
    assert turned.builds_its_own_encoder is False


def test_the_last_layer_named_is_the_standard_setting() -> None:
    assert EncoderSetting.standard().tuned("layer", "last") == EncoderSetting.standard()


@pytest.mark.parametrize(
    ("value", "reading", "reads_tokens"),
    [("beside", SummaryReading.BESIDE, True), ("only", SummaryReading.ONLY, False)],
)
def test_the_summaries_turn_and_are_named_only_when_turned(
    value: str, reading: SummaryReading, reads_tokens: bool
) -> None:
    turned = EncoderSetting.standard().tuned("summaries", value)

    assert turned.summaries is reading
    assert turned.reads_summaries
    assert turned.reads_tokens is reads_tokens
    assert turned.turned_away() == {"summaries": value}
    assert turned.builds_its_own_encoder is False


def test_the_standard_setting_reads_no_summaries_and_runs_the_encoder() -> None:
    standard = EncoderSetting.standard()

    assert standard.summaries is SummaryReading.NONE
    assert not standard.reads_summaries
    assert standard.reads_tokens
    assert standard.tuned("summaries", "none") == standard


def test_bins_lay_the_window_out_as_cells_and_build_the_encoder_anew() -> None:
    binned = EncoderSetting.standard().tuned("bins", "12")

    assert binned.bins == 12
    assert binned.reads_cells
    assert binned.cells is CellSet.WINDOW
    assert binned.builds_its_own_encoder
    assert binned.turned_away() == {"bins": 12}
    assert not EncoderSetting.standard().reads_cells


def test_the_cell_set_turns_only_over_bins_and_is_named_only_when_turned() -> None:
    binned = EncoderSetting.standard().tuned("bins", "6")

    over_corpus = binned.tuned("cells", "corpus")

    assert over_corpus.cells is CellSet.CORPUS
    assert over_corpus.turned_away() == {"bins": 6, "cells": "corpus"}
    assert binned.tuned("cells", "window") == binned
    with pytest.raises(UnknownKnobError, match="knob of the bins"):
        EncoderSetting.standard().tuned("cells", "corpus")
    with pytest.raises(InvalidEncoderSettingError, match="knob of the bins"):
        EncoderSetting(cells=CellSet.CORPUS)


@pytest.mark.parametrize("value", ["0", "-3", "1.5", "twelve"])
def test_bins_that_are_not_a_positive_count_are_refused(value: str) -> None:
    with pytest.raises(UnknownKnobError):
        EncoderSetting.standard().tuned("bins", value)


def test_a_clip_bounds_a_reading_and_is_refused_over_cells() -> None:
    with pytest.raises(UnknownKnobError, match="a cell carries its values otherwise"):
        EncoderSetting.standard().tuned("bins", "12").tuned("value_clip", "3")
    with pytest.raises(InvalidEncoderSettingError, match="a cell carries its values otherwise"):
        EncoderSetting(bins=12, value_clip=3.0)


def test_bins_and_a_grid_are_two_layouts_and_a_setting_takes_one() -> None:
    with pytest.raises(UnknownKnobError, match="two layouts"):
        EncoderSetting.standard().tuned("bins", "12").tuned("grid_resolution", "1")
    with pytest.raises(UnknownKnobError, match="two layouts"):
        EncoderSetting.standard().tuned("grid_resolution", "1").tuned("bins", "12")


def test_the_learnt_time_encoding_builds_the_encoder_anew_and_is_named_only_when_turned() -> None:
    learnt = EncoderSetting.standard().tuned("time_encoding", "learned")

    assert learnt.time_encoding is TimeEncoding.LEARNED
    assert learnt.builds_its_own_encoder
    assert learnt.turned_away() == {"time_encoding": "learned"}
    assert EncoderSetting.standard().tuned("time_encoding", "fourier") == EncoderSetting.standard()
    with pytest.raises(UnknownKnobError):
        EncoderSetting.standard().tuned("time_encoding", "sinusoidal")
