import pytest

from emblema.evaluation.adapters.detection.feature_routed_detection_runtime import (
    FeatureRoutedDetectionRuntime,
)
from emblema.evaluation.adapters.in_memory.detection_runtime import InMemoryDetectionRuntime
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.encoder_reading import EncoderReading
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.exceptions import UnsupportedDetectorFeaturesError
from tests.evaluation.support import WEIGHTS, labelled, task, window

FITTING = (window("a", 0, 10.0),)
SCORED = (labelled("c", 2, 10.0, 1.0),)
RECIPES = {
    "cells": DetectorRecipe(
        features=DetectorFeatures.RAW_CELLS,
        reading=None,
        neighbours=5,
        contamination=0.02,
        seed=1,
    ),
    "states": DetectorRecipe(
        features=DetectorFeatures.ENCODER_STATES,
        reading=EncoderReading(architecture=WEIGHTS, backbone=WEIGHTS),
        neighbours=5,
        contamination=0.02,
        seed=1,
    ),
    "summaries": DetectorRecipe(
        features=DetectorFeatures.READING_COUNTS,
        reading=None,
        neighbours=5,
        contamination=0.02,
        seed=1,
    ),
    "pretext": DetectorRecipe(
        features=DetectorFeatures.LINEAR_ANSWER_ERROR,
        reading=None,
        neighbours=None,
        contamination=0.02,
        seed=1,
        pretext=PretextAsking(backbone=WEIGHTS),
    ),
}


# The cells go where the states go: the runtime that lays windows out on the backbone's spans.
ROUTES = {"cells": "states", "states": "states", "summaries": "summaries", "pretext": "pretext"}


@pytest.mark.parametrize("route", list(RECIPES))
def test_a_recipe_goes_to_the_runtime_its_features_need(route: str) -> None:
    runtimes = {name: InMemoryDetectionRuntime() for name in ROUTES.values()}
    routed = FeatureRoutedDetectionRuntime(
        states=runtimes["states"], summaries=runtimes["summaries"], pretext=runtimes["pretext"]
    )

    routed.fit(RECIPES[route], task(), FITTING, SCORED, retain=False)

    assert {name: len(runtime.detections) for name, runtime in runtimes.items()} == {
        name: int(name == ROUTES[route]) for name in runtimes
    }


def test_a_process_that_asks_no_pretext_refuses_a_detector_that_would() -> None:
    routed = FeatureRoutedDetectionRuntime(
        states=InMemoryDetectionRuntime(), summaries=InMemoryDetectionRuntime()
    )

    with pytest.raises(UnsupportedDetectorFeaturesError, match="asks no backbone's pretext"):
        routed.fit(RECIPES["pretext"], task(), FITTING, SCORED, retain=False)
