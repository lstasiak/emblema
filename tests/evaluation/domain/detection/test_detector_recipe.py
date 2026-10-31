import pytest

from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_distance import ScoreDistance
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


ASKING = PretextAsking(backbone=WEIGHTS)


@pytest.mark.parametrize(
    "features",
    [
        DetectorFeatures.PRETEXT_ERROR,
        DetectorFeatures.LINEAR_ANSWER_ERROR,
        DetectorFeatures.INTERPOLATION_ERROR,
    ],
)
def test_a_detector_scored_by_an_error_names_the_masks_and_no_neighbours(
    features: DetectorFeatures,
) -> None:
    asked = recipe(features=features, reading=None, neighbours=None, pretext=ASKING)

    assert asked.pretext is ASKING
    assert features.asks_the_pretext
    assert not features.reads_an_encoder
    with pytest.raises(InvalidDetectorRecipeError, match="scores by an error"):
        asked.nearest()


def test_masks_named_or_missing_against_the_features_are_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="names no masks"):
        recipe(features=DetectorFeatures.PRETEXT_ERROR, reading=None, neighbours=None)
    with pytest.raises(InvalidDetectorRecipeError, match="names masks it does not ask"):
        recipe(pretext=ASKING)


def test_neighbours_named_or_missing_against_the_score_are_refused() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="reads no neighbours"):
        recipe(features=DetectorFeatures.PRETEXT_ERROR, reading=None, pretext=ASKING)
    with pytest.raises(InvalidDetectorRecipeError, match="names no neighbours"):
        recipe(neighbours=None)


def test_a_distance_reads_the_neighbours_it_names() -> None:
    assert recipe(neighbours=3).nearest() == 3


def test_a_window_is_hidden_at_least_once() -> None:
    assert ASKING.draws == 8
    with pytest.raises(InvalidDetectorRecipeError, match="at least once"):
        PretextAsking(backbone=WEIGHTS, draws=0)


def test_the_masks_are_read_off_a_detector_that_asks_the_pretext_alone() -> None:
    asked = recipe(
        features=DetectorFeatures.PRETEXT_ERROR, reading=None, neighbours=None, pretext=ASKING
    )

    assert asked.asked() is ASKING
    with pytest.raises(InvalidDetectorRecipeError, match="reads features, not the pretext"):
        recipe().asked()


def test_the_whitened_distance_reads_no_neighbours() -> None:
    whitened = recipe(distance=ScoreDistance.WHITENED, neighbours=None)

    assert not whitened.scores_by_nearest_neighbours
    assert recipe().scores_by_nearest_neighbours
    with pytest.raises(InvalidDetectorRecipeError, match="reads no neighbours"):
        whitened.nearest()
    with pytest.raises(InvalidDetectorRecipeError, match="reads no neighbours"):
        recipe(distance=ScoreDistance.WHITENED)
    with pytest.raises(InvalidDetectorRecipeError, match="names no neighbours"):
        recipe(neighbours=None)


def test_a_detector_scored_by_an_error_has_no_distance_to_choose() -> None:
    with pytest.raises(InvalidDetectorRecipeError, match="no distance to choose"):
        recipe(
            features=DetectorFeatures.PRETEXT_ERROR,
            reading=None,
            neighbours=None,
            pretext=PretextAsking(backbone=WEIGHTS, draws=2),
            distance=ScoreDistance.WHITENED,
        )


def test_only_a_detector_scored_by_an_error_reads_its_errors_per_channel() -> None:
    asked = recipe(
        features=DetectorFeatures.PRETEXT_ERROR,
        reading=None,
        neighbours=None,
        pretext=PretextAsking(backbone=WEIGHTS, draws=2),
        normalisation=ErrorNormalisation.PER_CHANNEL,
    )

    assert asked.normalisation is ErrorNormalisation.PER_CHANNEL
    assert recipe().normalisation is ErrorNormalisation.RAW
    with pytest.raises(InvalidDetectorRecipeError, match="no errors to read"):
        recipe(normalisation=ErrorNormalisation.PER_CHANNEL)
