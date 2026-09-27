import pytest

from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.spectral_features import SpectralFeatures
from emblema.evaluation.adapters.features.window_features import WindowFeatures, features_for
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme


@pytest.mark.parametrize(
    ("scheme", "reading"),
    [
        (FeatureScheme.PER_CHANNEL, PerChannelFeatures(3)),
        (FeatureScheme.SPECTRAL, SpectralFeatures(3)),
        (FeatureScheme.CHANNEL_AGGREGATED, ChannelAggregatedFeatures()),
    ],
)
def test_every_scheme_names_one_reading_as_wide_as_the_corpus(
    scheme: FeatureScheme, reading: WindowFeatures
) -> None:
    assert features_for(scheme, 3).names() == reading.names()
