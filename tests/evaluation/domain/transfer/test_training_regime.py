import pytest

from emblema.evaluation.domain.exceptions import InvalidTrainingRegimeError, UnknownKnobError
from emblema.evaluation.domain.transfer.training_regime import ClassWeight, TrainingRegime


def test_the_standard_regime_runs_every_epoch_weighs_nothing_and_reads_every_channel() -> None:
    standard = TrainingRegime.standard()

    assert standard.stops is False
    assert standard.stop_partly_stated is False
    assert standard.class_weight is ClassWeight.NONE
    assert standard.channel_dropout == 0.0
    assert standard.turned_away() == {}
    assert standard.parameters() == {
        "stop_share": 0.0,
        "patience": 0,
        "class_weight": "none",
        "channel_dropout": 0.0,
    }


def test_the_stop_is_whole_once_its_two_knobs_are_turned_one_at_a_time() -> None:
    half = TrainingRegime.standard().tuned("patience", "10")

    assert half.stop_partly_stated
    assert not half.stops
    whole = half.tuned("stop_share", "0.2")
    assert whole.stops
    assert not whole.stop_partly_stated
    assert whole.turned_away() == {"stop_share": 0.2, "patience": 10}


def test_the_weight_and_the_channel_dropout_turn_and_are_named_only_when_turned() -> None:
    turned = (
        TrainingRegime.standard().tuned("class_weight", "ratio").tuned("channel_dropout", "0.2")
    )

    assert turned.class_weight is ClassWeight.RATIO
    assert turned.turned_away() == {"class_weight": "ratio", "channel_dropout": 0.2}
    assert turned.parameters()["class_weight"] == "ratio"


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("stop_share", "0.6"),
        ("stop_share", "-0.1"),
        ("stop_share", "nan"),
        ("patience", "-1"),
        ("patience", "ten"),
        ("class_weight", "balanced"),
        ("channel_dropout", "1"),
        ("channel_dropout", "inf"),
        ("early_stop", "yes"),
    ],
)
def test_a_knob_the_regime_has_not_or_a_value_it_cannot_take_is_refused(
    knob: str, value: str
) -> None:
    with pytest.raises(UnknownKnobError):
        TrainingRegime.standard().tuned(knob, value)


def test_the_invariants_hold_at_construction_too() -> None:
    with pytest.raises(InvalidTrainingRegimeError, match="stop_share"):
        TrainingRegime(stop_share=0.7, patience=3)
    with pytest.raises(InvalidTrainingRegimeError, match="channel_dropout"):
        TrainingRegime(channel_dropout=1.0)
