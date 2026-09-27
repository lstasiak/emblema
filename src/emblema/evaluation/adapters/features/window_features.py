"""What a classical candidate reads a window as, whichever scheme it was stated under.

A closed union rather than a protocol: the schemes are a closed enum, so the readings of a
window are a closed set too, and naming them here lets whoever holds one be typed without
inventing an interface that has exactly these implementations and no others.
"""

from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.spectral_features import SpectralFeatures
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme

type WindowFeatures = PerChannelFeatures | SpectralFeatures | ChannelAggregatedFeatures


def features_for(scheme: FeatureScheme, channels: int) -> WindowFeatures:
    """The reading of a window ``scheme`` names, as wide as ``channels`` where width matters.

    Decided once: the runtime that fits the trees and the service that answers from them must
    read a window the same way, or the trees are fed columns they were never fitted on.
    """
    match scheme:
        case FeatureScheme.PER_CHANNEL:
            return PerChannelFeatures(channels)
        case FeatureScheme.SPECTRAL:
            return SpectralFeatures(channels)
        case FeatureScheme.CHANNEL_AGGREGATED:
            return ChannelAggregatedFeatures()
