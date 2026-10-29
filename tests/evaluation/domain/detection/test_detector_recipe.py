import pytest

from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.exceptions import InvalidDetectorRecipeError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from tests.evaluation.support import WEIGHTS

READING = EncoderReading(architecture=WEIGHTS, backbone=WEIGHTS)


def recipe(**overrides: object) -> DetectorRecipe:
    stated = {
        "features": DetectorFeatures.ENCODER_STATES,
        "reading": READING,
        "neighbours": 5,
        "contamination": 0.02,
        "seed": 1,
    }
    stated.update(overrides)
    return DetectorRecipe(**stated)  # type: ignore[arg-type]


def test_a_detector_over_an_encoders_states_names_the_encoder_and_one_over_summaries_none() -> None:
    assert recipe().reading is READING
    assert recipe(features=DetectorFeatures.SUMMARIES, reading=None).reading is None
    assert DetectorFeatures.ENCODER_STATES.reads_an_encoder
    assert not DetectorFeatures.SUMMARIES.reads_an_encoder


def test_an_encoder_named_or_missing_against_the_features_is_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="names no encoder"):
        recipe(reading=None)
    with pytest.raises(InvalidDetectorRecipeError, match="does not read"):
        recipe(features=DetectorFeatures.SUMMARIES)


def test_an_encoder_at_its_initialisation_names_its_shape_and_no_weights() -> None:
    untrained = EncoderReading(architecture=WEIGHTS, backbone=None)

    assert untrained.backbone is None
    assert untrained.encoder == EncoderSetting.standard()


def test_a_reading_that_would_learn_is_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="cannot pool"):
        EncoderReading(
            architecture=WEIGHTS,
            backbone=WEIGHTS,
            pooling=HeadPooling(pooling=PoolingScheme.ATTENTION),
        )
    with pytest.raises(InvalidDetectorRecipeError, match="drops nothing"):
        EncoderReading(
            architecture=WEIGHTS,
            backbone=WEIGHTS,
            encoder=EncoderSetting.standard().tuned("dropout", "0.1"),
        )


@pytest.mark.parametrize(
    ("knob", "value", "reason"),
    [
        ("grid_resolution", "4", "neither gridded"),
        ("value_clip", "3", "neither gridded"),
        ("summaries", "beside", "no summaries"),
    ],
)
def test_a_reading_the_detector_would_not_apply_is_refused(
    knob: str, value: str, reason: str
) -> None:
    with pytest.raises(InvalidDetectorRecipeError, match=reason):
        EncoderReading(
            architecture=WEIGHTS,
            backbone=None,
            encoder=EncoderSetting.standard().tuned(knob, value),
        )


def test_pretrained_weights_fix_the_encoders_build() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="fix the encoder's build"):
        EncoderReading(
            architecture=WEIGHTS,
            backbone=WEIGHTS,
            encoder=EncoderSetting.standard().tuned("bins", "6"),
        )
    # The shape at its initialisation may be read over cells, as the pretrained one is.
    EncoderReading(
        architecture=WEIGHTS, backbone=None, encoder=EncoderSetting.standard().tuned("bins", "6")
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"neighbours": 0}, "at least one neighbour"),
        ({"contamination": 0.0}, "strictly between"),
        ({"contamination": 1.0}, "strictly between"),
        ({"contamination": float("nan")}, "strictly between"),
    ],
    ids=["neighbours", "no contamination", "all contamination", "nan"],
)
def test_the_neighbours_and_the_contamination_are_bounded(
    overrides: dict[str, object], reason: str
) -> None:
    with pytest.raises(InvalidDetectorRecipeError, match=reason):
        recipe(**overrides)
