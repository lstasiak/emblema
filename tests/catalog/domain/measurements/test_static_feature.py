import pytest

from emblema.catalog.domain.exceptions import InvalidStaticFeatureError
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from tests.catalog.domain.support import BLANK_OR_PADDED, NON_FINITE


@pytest.mark.parametrize("channel", BLANK_OR_PADDED)
def test_static_feature_rejects_blank_or_padded_channel(channel: str) -> None:
    with pytest.raises(InvalidStaticFeatureError, match="non-blank"):
        StaticFeature(channel, 1.0)


@pytest.mark.parametrize("value", NON_FINITE)
def test_static_feature_value_is_finite(value: float) -> None:
    with pytest.raises(InvalidStaticFeatureError, match="finite"):
        StaticFeature("age", value)
