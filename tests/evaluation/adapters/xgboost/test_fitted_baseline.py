import io

import joblib
import pytest
import xgboost

from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from tests.evaluation.support import recipe

FEATURES = ChannelAggregatedFeatures()
ROWS = [[float(row + column) for column in range(FEATURES.width)] for row in range(6)]
TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def grown() -> xgboost.XGBRegressor:
    model = xgboost.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=1)
    model.fit(ROWS, TARGETS)
    return model


def kept() -> FittedBaseline:
    return FittedBaseline.of(recipe(), grown(), feature_names=FEATURES.names(), target_scale=125.0)


def test_a_fitted_baseline_reads_back_as_what_was_written() -> None:
    read = FittedBaseline.read(kept().to_bytes())

    assert read.target_scale == 125.0
    assert read.feature_names == FEATURES.names()
    assert read.parameters == recipe().parameters()


def test_the_trees_that_come_back_answer_what_the_fitted_model_answered() -> None:
    model = grown()
    stored = FittedBaseline.of(recipe(), model, feature_names=FEATURES.names(), target_scale=1.0)

    read = FittedBaseline.read(stored.to_bytes())

    assert read.booster().inplace_predict(ROWS) == pytest.approx(model.predict(ROWS))


@pytest.mark.parametrize(
    ("named", "content"),
    [
        ("nothing", b""),
        ("prose", b"not a model at all"),
        ("zeros", bytes(64)),
        ("an archive", b"PK\x03\x04" + bytes(20)),
    ],
)
def test_bytes_that_are_not_a_fitted_baseline_are_refused(named: str, content: bytes) -> None:
    with pytest.raises(UnreadableFittedCandidateError, match="not a fitted baseline"):
        FittedBaseline.read(content)


def test_a_document_missing_what_this_reads_is_refused() -> None:
    buffer = io.BytesIO()
    joblib.dump({"parameters": {}, "target_scale": 1.0}, buffer)

    with pytest.raises(UnreadableFittedCandidateError, match="not a fitted baseline"):
        FittedBaseline.read(buffer.getvalue())


def test_trees_that_xgboost_will_not_load_are_refused() -> None:
    broken = FittedBaseline(parameters={}, feature_names=(), target_scale=1.0, model=b"not trees")

    with pytest.raises(UnreadableFittedCandidateError, match="not trees"):
        broken.booster()
